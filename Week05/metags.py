#!/usr/bin/env python3
import argparse
import os
import platform
import subprocess
import sys
import tempfile

if platform.python_version_tuple() >= ("3", "12"):
    from itertools import batched  # type: ignore


def int_to_script(n: int, /) -> str:
    """Converts a Meta GolfScript integer into the GolfScript Program"""
    s = format(n, "x")

    if (len(s) % 2) != 0:
        s = "0" + s
    if platform.python_version_tuple() >= ("3", "12"):
        return "".join(chr(int("".join(i), base=16) - 1) for i in batched(s, 2))
    else:
        return "".join(
            chr(int(s[i : i + 2], base=16) - 1) for i in range(0, len(s), 2)
        )


def script_to_int(script: str, /) -> int:
    """Converts GolfScript code into the Meta GolfScript integer"""
    return int("".join(format(ord(c) + 1, "0>2x") for c in script), base=16)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Meta GolfScript Helper",
        description="A script to convert GolfScript code into a Meta GolfScript program, and vice versa",
    )

    direction_group = parser.add_mutually_exclusive_group(required=True)

    direction_group.add_argument(
        "-i",
        "-I",
        "--int",
        action="store_true",
        help="Takes and integer and converts in to GolfScript",
    )
    direction_group.add_argument(
        "-s",
        "-S",
        "--script",
        action="store_true",
        help="Takes GolfScript code and converts it to a Meta GolfScript integer",
    )

    parser.add_argument(
        "--run",
        "-r",
        help="Runs the resulting script, in addition to printing it, does nothing with -i",
        action="store_true",
    )

    # io_group=parser.add_argument_group('I/O',description='Handles input and output')
    # io_group.add_argument('--input')
    # io_group.add_argument('--output')
    return parser.parse_args()


def main():
    args = parse_args()

    if sys.stdin.isatty():  # if interactive
        var = input()
    else:  # if redirecting from file, this allows newlines to be read
        var = sys.stdin.read()

    sys.set_int_max_str_digits(0)
    if args.int:
        script = int_to_script(int(var.strip()))
        print(script)
        if args.run:
            with tempfile.TemporaryDirectory() as dir:  # create a temporary dir to write the file, if upgrade to 3.12 can use delete_on_close param
                with open(os.path.join(dir, "tmp.tmp"), "w") as f:
                    f.write(script)

                subprocess.run(["./golfscript.rb", f.name])
    elif args.script:
        n = script_to_int(var)

        print(n)


if __name__ == "__main__":
    main()
