"""PyInstaller entry point; divert spawned workers before importing the GUI."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    from arte_profiler.gui.packaged import main

    main()
