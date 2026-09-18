"""LightGBM entry point (historical module spelling retained)."""
from utils.cv_runner import parse_args as _parse_args, run_cv


def parse_args(argv=None):
    return _parse_args('LightGBM', argv)


def main():
    return run_cv('LightGBM', parse_args())


if __name__ == '__main__':
    main()
