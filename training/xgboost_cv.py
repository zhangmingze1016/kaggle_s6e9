"""XGBoost CV entry point; legacy CLI flags remain supported."""
from utils.cv_runner import parse_args as _parse_args, run_cv


def parse_args(argv=None):
    return _parse_args('XGBoost', argv)


def main():
    return run_cv('XGBoost', parse_args())


if __name__ == '__main__':
    main()
