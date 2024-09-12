#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from lipmad import project
from lipmad import base_conf

log = base_conf.log


def main():
    """ Initialize a new project """
    description = """ Create a new project. """
    epilog = """
            Examples: Usage example goes here.

            Notes:

            """
    # Create argument parser
    parser = argparse.ArgumentParser(prog='lipmad-create_project',
                                     epilog=epilog, description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional mandatory argument
    parser.add_argument("project", type=str,
                        help="Directory containing sets of aerial image(s).")

    args = parser.parse_args()

    if not os.path.isdir(args.project):
        log.error(f"Input argument is not a directory: {args.project}")
        quit()
    else:
        log.info(f"Project directory: {args.project}")

    # create an empty project
    proj = project.ProjectMgr(input_arg=args.project, create=True)

    # save project
    proj.save()


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
