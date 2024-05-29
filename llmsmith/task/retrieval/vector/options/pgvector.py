from typing import Literal, TypedDict

try:
    from sqlalchemy.sql import ColumnElement
except ImportError:
    raise ImportError(
        "'sqlalchemy', 'psycopg' and 'pgvector' libraries are required to use PgVector. You can install it with `pip install \"llmsmith[pgvector]\"`"
    )


class PgVectorQueryOptions(TypedDict):
    """
    A dictionary of options to pass while querying a PgVector backed table.

    :param limit: number of documents to fetch
    :type name: int
    :param where: filters to be applied in the query. Use Sqlalchemy where clauses
    :type where: str
    :param distance_function: distance function to use. Can be 'l2' or 'cosine'. Defaults to 'cosine'
    :type distance_function: str
    """

    limit: int
    where: ColumnElement[bool]
    distance_function: Literal["l2", "cosine"]


def _query_options_dict(options: PgVectorQueryOptions) -> dict:
    opt = {attr: options.get(attr) for attr in PgVectorQueryOptions.__annotations__}

    if not opt.get("limit"):
        opt["limit"] = 10

    if not opt.get("distance_function"):
        opt["distance_function"] = "cosine"

    return opt
