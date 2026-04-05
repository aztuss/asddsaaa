import argparse
import threading
import time

from data.auto_data import ensure_all_data, refresh_all_data


def data_loop():
    ensure_all_data(force=False)
    while True:
        time.sleep(3600)
        refresh_all_data()


def cmd_run(broker: str):
    t = threading.Thread(target=data_loop, daemon=True)
    t.start()

    from core.scanner import MarketScanner
    scanner = MarketScanner()

    scanner.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", choices=["demo", "simplefx"], default="demo")
    args = parser.parse_args()

    cmd_run(args.broker)


if __name__ == "__main__":
    main()
