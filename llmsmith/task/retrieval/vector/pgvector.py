import logging
from typing import Callable, List

from sqlalchemy import Select
from sqlalchemy.sql.elements import BindParameter

from llmsmith.reranker.base import Reranker
from llmsmith.task.base import Task
from llmsmith.task.models import TaskInput, TaskOutput
from llmsmith.task.retrieval.vector.base import EmbeddingFunc, default_doc_processor
from llmsmith.task.retrieval.vector.options.pgvector import (
    PgVectorQueryOptions,
    _query_options_dict,
)


try:
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy import text
    from sqlalchemy.engine.cursor import CursorResult
    from sqlalchemy.sql import ColumnElement
    from sqlalchemy.sql.expression import select, table
except ImportError:
    raise ImportError(
        "'sqlalchemy', 'psycopg' and 'pgvector' libraries are required to use PgVector. You can install it with `pip install \"llmsmith[pgvector]\"`"
    )


log = logging.getLogger(__name__)

# Default options for querying a PgVector supported table.
default_options: PgVectorQueryOptions = PgVectorQueryOptions(limit=10)
supported_distance_functions = ["l2", "cosine"]


class PgVectorRetriever(Task[str, str]):
    """
    Task for retrieving documents from a table in Postgres DB (with PgVector extension).

    :param name: The name of the task.
    :type name: str
    :param db_engine: Sqlalchemy async engine object
    :type db_engine: :class:`sqlalchemy.ext.asyncio.AsyncEngine`
    :param table_name: table where the embeddings are stored
    :type table_name: str
    :param text_colname: name of the column where embeddings are stored
    :type text_colname: str
    :param embedding_colname: name of the column where the actual text is stored
    :type embedding_colname: str
    :param embedding_func: Embedding function
    :type embedding_func: :class:`llmsmith.task.retrieval.vector.base.EmbeddingFunc`
    :param doc_processing_func: The function to process the query result, defaults to `llmsmith.task.retrieval.vector.base.default_doc_processor`.
    :type doc_processing_func: Callable[[List[str]], str], optional
    :param query_options: A dictionary of options to be used for querying PgVector table.
    :type query_options: :class:`llmsmith.task.retrieval.vector.options.pgvector.PgVectorQueryOptions`, optional
    :param reranker: Rerank the documents based on the query used to retrieve the documents.
    :type reranker: :class:`llmsmith.reranker.base.Reranker`, optional
    """

    def __init__(
        self,
        name: str,
        db_engine: AsyncEngine,
        table_name: str,
        text_colname: str,
        embedding_colname: str,
        embedding_func: EmbeddingFunc,
        doc_processing_func: Callable[[List[str]], str] = default_doc_processor,
        query_options: PgVectorQueryOptions = default_options,
        reranker: Reranker = None,
    ) -> None:
        super().__init__(name)

        if not embedding_func:
            raise ValueError("Embedding function ('embedding_func') is required")
        if not table_name:
            raise ValueError("'table_name' is required")
        if not text_colname:
            raise ValueError("'text_colname' is required")
        if not embedding_colname:
            raise ValueError("'embedding_colname' is required")

        self._db_engine = db_engine
        self._table_name = table_name
        self._text_column = text_colname
        self._embedding_column = embedding_colname
        self.embedding_func = embedding_func
        self.doc_processing_func = doc_processing_func
        self.query_options = query_options
        self._reranker = reranker

    async def execute(self, task_input: TaskInput[str]) -> TaskOutput[str]:
        """
        Executes the task of retrieving documents from the PgVector backed table.

        :param task_input: The input for the task.
        :type task_input: :class:`llmsmith.task.models.TaskInput[str]`
        :return: The output of the task, which includes the processed result and the raw output from psycopg.
        :rtype: :class:`llmsmith.task.models.TaskOutput[str]`
        """
        query_options: dict = _query_options_dict(self.query_options)

        log.debug(
            f"PgVector query request: input string: {task_input.content}\n OPTIONS: {query_options}"
        )

        embeddings = self.embedding_func([task_input.content])

        dist_func = query_options.get("distance_function")
        if dist_func not in supported_distance_functions:
            raise ValueError("distance_function only supports 'l2' or 'cosine'")
        dist_func_op = "<=>" if dist_func == "cosine" else "<->"

        stmt: Select = select(text("*")).select_from(table(self._table_name))

        # apply where clauses if any
        if query_options.get("where") is not None:
            filters: ColumnElement[bool] = query_options.get("where")
            stmt = stmt.filter(filters)

        stmt = stmt.order_by(
            text(f"{self._embedding_column} {dist_func_op} :embedding_val").bindparams(
                BindParameter(key="embedding_val", value=embeddings[0], type_=Vector)
            )
        ).limit(query_options.get("limit"))

        docs: List[str] = []
        async with self._db_engine.connect() as conn:
            rows: CursorResult = await conn.execute(stmt)
            records = [row._asdict() for row in rows]
            docs = [rec.get(self._text_column) for rec in records]

        if self._reranker and docs:
            docs = await self._reranker.rerank(query=task_input.content, docs=docs)

        processed_res: str = self.doc_processing_func(docs)
        return TaskOutput(content=processed_res, raw_output=records)
