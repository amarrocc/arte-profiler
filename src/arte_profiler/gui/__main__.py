"""Launch the local GUI with ``python -m arte_profiler.gui``."""


def main() -> None:
    """Open the GUI in a browser and listen only on the local machine."""
    try:
        from nicegui import ui
    except ModuleNotFoundError as error:
        if error.name != "nicegui":
            raise
        raise SystemExit(
            'The GUI requires NiceGUI. Install it with: '
            'python -m pip install -e ".[gui]"'
        ) from None

    from .app import create_page

    ui.page("/")(create_page)
    ui.run(
        host="127.0.0.1",
        port=8080,
        title="Arte-Profiler",
        show=True,
        native=False,
        reload=False,
    )


if __name__ == "__main__":
    main()
