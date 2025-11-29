import functools
import os

from rich.console import Console

CITATION_LINK = "https://github.com/BrainLesion/GlioMODA#citation"


def citation_reminder(func):
    """
    Decorator to remind users to cite gliomoda.

    The reminder is shown when the environment variable
    `GLIOMODA_CITATION_REMINDER` is set to "true" (default).
    To disable the reminder, set the environment variable to "false".

    Environment variable used:
    - GLIOMODA_CITATION_REMINDER: Controls whether the reminder is shown.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("GLIOMODA_CITATION_REMINDER", "true").lower() == "true":
            console = Console()
            console.rule("Thank you for using [bold]gliomoda[/bold]")
            console.print(
                "Please support our development by citing",
                justify="center",
            )
            console.print(
                f"{CITATION_LINK} -- Thank you!",
                justify="center",
            )
            console.rule()
            console.line()
        return func(*args, **kwargs)

    return wrapper
