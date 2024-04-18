class MaxTurnsReachedException(Exception):
    """Raised when the number of turns is exhausted while executing an agent."""

    def __init__(self):
        super().__init__("Reached maximum number of turns")
