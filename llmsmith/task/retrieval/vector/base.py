from typing import Callable, List, Union

# Type alias for embedding function
EmbeddingFunc = Callable[[List[str]], List[List[Union[float, int]]]]
