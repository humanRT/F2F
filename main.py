"""Start F2F with: python ./main.py (in the SAM3 environment)."""
import sys
import warnings

# Silence only timm's known legacy-import notice, not other runtime warnings.
warnings.filterwarnings(
    "ignore",
    message=r"^Importing from timm\.models\.layers is deprecated, please import via timm\.layers$",
    category=FutureWarning,
    module=r"^timm\.models\.layers$",
)

sys.dont_write_bytecode = True


def main(argv=None):
    """Launch camera capture and overlays, or forward explicit CLI arguments."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--sam3-worker"]:
        from pipe_edges.sam3_worker import serve
        serve()
        return 0
    from pipe_edges.cli import main as run_app
    if not arguments:
        arguments = ["camera", "--show"]
    return run_app(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
