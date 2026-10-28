# Experimental macOS browser application

This is a local packaging prototype, not a signed/notarized public release.
Python, NiceGUI, and scientific dependencies are included in a one-directory
`.app`. The browser is the interface; all processing stays on this computer.
The scientific implementation and CLI are unchanged.

## Build

Use **Python 3.11 on macOS**, with the architecture you intend to test. The
current prototype was built on Apple Silicon (`arm64`); it is not universal.
The repository's macOS ArgyllCMS executables are **Intel `x86_64`**, so Apple
Silicon requires Rosetta 2. The packaging does not install or replace ArgyllCMS
or Rosetta. An Intel Python environment is needed to produce an Intel app.

From the repository root:

```sh
python3.11 -m venv .venv-macos
source .venv-macos/bin/activate
python -m pip install -r packaging/macos/requirements.txt
python -m pip install --no-deps --no-build-isolation -e ".[gui]"
python packaging/macos/build.py
```

The exact build command after environment setup is
`python packaging/macos/build.py`. It runs PyInstaller with the committed spec,
cleans its cache, and replaces previous prototype outputs. Allow several GB of
free space for dependencies, working files, both output layouts, and workflow
results. Build products are ignored by Git:

- **Double-clickable application:** `dist/macos/Arte-Profiler.app`
- Console-debuggable directory layout: `dist/macos/Arte-Profiler/`
- Actual build environment: `dist/macos/build-environment.txt`
- Working files and PyInstaller diagnostics: `build/macos/Arte-Profiler/`

Keep the whole `.app` intact. Finder treats it as one application. You can copy
it to another folder without installing Python. The companion directory is
useful for debugging but is not needed alongside the `.app`.

`requirements.txt` records the prototype's runtime/build versions rather than
changing the package's normal dependency ranges. It is a version snapshot, not
a claim of byte-for-byte reproducibility or compatibility with every macOS
version. If the environment changes, retain `build-environment.txt` with test
results. Use the same dependency pins and architecture to repeat a build.

## Launch and stop

1. Double-click `dist/macos/Arte-Profiler.app` in Finder. It opens the system's
   default browser at `http://127.0.0.1:<available-port>/`. It uses an available
   port rather than competing for the development server's port 8080.
2. Use the existing workflow forms and select a writable **Results folder**.
   Generated ICC/PDF files, intermediate images, measurements and
   `profiling.log` go there. Selecting a folder inside the app is rejected.
3. Use **Quit Arte-Profiler** above the interface and confirm **Quit** to stop
   the server. Quit is refused during processing; wait for completion or a
   reported failure. Closing the browser tab alone does not stop the app.
4. If you close the tab, double-click the app again to reopen the running
   instance. An OS lock prevents two packaged instances from starting jobs.

Startup checks locate required resources and run each bundled ArgyllCMS tool's
help command. A startup failure displays a macOS alert pointing to
`~/Library/Logs/Arte-Profiler/startup.log`. Startup and worker console messages
also go there when launched without a terminal. This is separate from the
scientific `profiling.log` in the chosen results folder. Caches and server
storage go to `~/Library/Caches/Arte-Profiler/`; temporary scientific files use
the existing backend's results-folder or system temporary-file handling.
The bundle is never used as a working/output directory.
Browse locations are saved separately in
`~/Library/Application Support/Arte-Profiler/browse.json`, with independent input
and results-folder histories. Cancel keeps the last successfully visited location;
relaunching or rebuilding the app does not reset it.

For troubleshooting, start the executable from Terminal:

```sh
"dist/macos/Arte-Profiler.app/Contents/MacOS/Arte-Profiler"
```

The last running URL and PID are in
`~/Library/Caches/Arte-Profiler/session.json`. The file is removed on normal
shutdown. If the browser interface is unavailable, Activity Monitor can locate
Arte-Profiler by this PID and quit it. Force Quit can interrupt processing and
leave partial results; use the interface's Quit action normally.

## Automated checks

With pytest, pytest-asyncio, and the development package installed:

```sh
python -m pytest tests/test_macos_packaging.py tests/test_gui_packaged.py tests/test_gui_smoke.py tests/test_gui_service.py tests/test_gui_controller.py -q
ARTE_PROFILER_PACKAGED_APP="$PWD/dist/macos/Arte-Profiler.app" python -m pytest tests/test_macos_packaging.py -q
codesign --verify --deep --strict dist/macos/Arte-Profiler.app
```

The opt-in frozen test runs the executable from another working directory with
an isolated home and a minimal system PATH, removing Python/Conda environment
variables. It verifies that both the server process and a **spawned frozen
worker** can import the scientific dependencies, resolve bundled target images,
CHT/CIE references, guidelines, report fonts, and NiceGUI assets, and execute
`scanin`, `colprof`, and `icclu`. It does not generate an ICC/PDF or exercise the
browser. It needs macOS and Rosetta where applicable.

You can also run the frozen diagnostic directly, using an absolute output path:

```sh
"dist/macos/Arte-Profiler.app/Contents/MacOS/Arte-Profiler" --smoke-test "$PWD/build/macos/smoke"
```

A successful run exits with code 0 and writes `smoke.json` and `worker.json` in
that directory. `--no-browser` starts the real server without opening a browser
for browser automation; it still requires an explicit Quit or a termination
signal when finished.

### Prototype validation on 2026-10-25

Executed on this Apple Silicon Mac with Rosetta:

- Installed the pinned requirements in a clean venv and built the `.app`.
- Full test suite with the frozen-resource test enabled: **61 passed, 1 skipped**
  (the separate opt-in source scientific smoke test). The final clean build's
  packaging checks also passed: **5 passed**.
- Headless Chrome against the final frozen app, with isolated HOME and minimal
  PATH: created an ICC and creation/evaluation PDFs from the dual-target TIFF,
  then evaluated the generated ICC against the separate DT-NGT2 TIFF. Verified
  the ICC header, three nonempty PDF files, processing logs, and browser results.
- Verified Quit refuses during processing, cancellation keeps the app running,
  confirmation exits with code 0, the session file is removed, and the old URL
  stops responding.
- Local signature verification passed. Checked 463 bundled Mach-O files with
  `otool -L`; no absolute non-system library dependencies were found.

Finder/default-browser launch, another physical Mac, research-network storage,
custom CIE files, and the remaining manual workflow combinations below still
require manual verification. Browser automation used `--no-browser`; it does
not establish that Launch Services opens the default browser on another Mac.

## Manual end-to-end verification

Use a new results folder for each run and enough free disk space. Do not use a
cloud placeholder TIFF; make the file available locally first.

1. Quit the development server, deactivate Conda/venvs, and launch the `.app`
   through Finder. Verify that the **default browser** opens and both workflow
   choices are available. No Terminal or editor should be needed.
2. **Creation:** select `examples/sample_CCSG.tiff`, `ColorCheckerSG`, and
   "I only have the profiling target." Review and Run. Verify live logs,
   `input_profile.icc` and `profile_creation_report.pdf`; open the PDF and check
   its title and plots. Check `profiling.log` in the selected folder.
3. **Same-TIFF evaluation:** create using `examples/sample_dual_capture.tiff`
   with `ColorCheckerSG`; choose another target in the same TIFF and `DT-NGT2`.
   Verify both PDFs and the ICC. The evaluation target must differ from the
   profiling target type.
4. **Separate-TIFF evaluation:** use `sample_CCSG.tiff` for creation and
   `sample_DT-NGT2.tiff` with `DT-NGT2` for evaluation. Verify both PDFs and ICC.
5. **Existing ICC:** evaluate the ICC from step 2 against
   `examples/sample_DT-NGT2.tiff` with `DT-NGT2` in a new results folder. Verify
   `profile_evaluation_report.pdf` without a newly generated ICC. Also check a
   custom CIE reference you normally use.
6. During a run, verify Quit refuses to stop processing. After completion,
   cancel Quit once, then confirm Quit. Verify the old URL stops responding,
   `session.json` disappears, and Activity Monitor shows no remaining
   Arte-Profiler worker. Relaunch and verify that a new session works.
7. Copy the intact `.app` to a location outside the repository. Repeat with
   inputs/results outside the repository, ideally on a second Mac without
   Python or Conda. Test ordinary double-clicking and the intended research
   network storage. Automated PATH isolation cannot prove operation on another
   physical Mac or every network filesystem.

## Resource and compatibility decisions

- The spec follows [NiceGUI's documented PyInstaller approach](https://nicegui.io/documentation/section_configuration_deployment#package_for_installation),
  including NiceGUI assets and `reload=False`. It uses a spec because this app
  also needs scientific resources, hidden codecs, and a worker runtime hook.
  Browser-mode windowless apps need an explicit quit action; this prototype
  supplies one instead of introducing a native window.
- `freeze_support()` runs before GUI imports, as required by
  [PyInstaller multiprocessing guidance](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#multi-processing).
  The runtime hook sets writable caches, standard streams, and the noninteractive
  plotting backend in both server and worker processes.
- The original `importlib.resources` lookups are preserved by mirroring the
  `arte_profiler` resource layout inside the bundle. The spec explicitly includes
  `guidelines.yaml`, which the current wheel package-data patterns omit. This
  packaging prototype does not change that separate wheel issue.
- Resources come from the target manifest, with the two report fonts, existing
  guidelines, licenses, and just the three macOS executables used by this API.
  Other platform binaries and untracked custom reference/profile experiments
  are excluded. No scientific data is rewritten.
- PyInstaller applies an **ad-hoc signature** for local execution. There is no
  Developer ID signature, notarization, installer, or promised Gatekeeper
  acceptance on another Mac. Existing ArgyllCMS binaries lack modern SDK
  metadata; PyInstaller warns about future hardened-runtime signing. Successful
  local code-signature verification is not notarization.
- Python extensions match the build architecture. The existing Intel ArgyllCMS
  tools remain Intel executables and depend on Rosetta on Apple Silicon. Their
  actual reported versions are recorded by the smoke test; the repository's
  `argyllcms_v3.3.0` directory name is not treated as a version guarantee.
- Large TIFFs retain the backend's existing memory, disk-space and processing
  requirements. Packaging does not fix scientific memory usage or unavailable
  cloud/network files. Minimum supported macOS and Intel compatibility require
  testing on the intended hardware; this milestone creates no other OS builds.
