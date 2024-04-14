#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
import logging
import sys
import traceback

from aicsimageio.writers import OmeTiffWriter
from lsfm_destripe import DeStripe, get_module_version

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################


def list_of_floats(arg):
    return list(map(float, arg.split(",")))


def bool_args(arg):
    if ("false" == arg) or ("False" == arg):
        return False
    elif ("true" == arg) or ("True" == arg):
        return True


class Args(argparse.Namespace):

    DEFAULT_FIRST = 10
    DEFAULT_SECOND = 20

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.first = self.DEFAULT_FIRST
        self.second = self.DEFAULT_SECOND
        self.debug = False
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            prog="run_destripe",
            description="run destripe for LSFM images",
        )

        p.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s " + get_module_version(),
        )

        p.add_argument(
            "--is_vertical",
            type=bool_args,
            required=True,
        )

        p.add_argument(
            "--angle_offset",
            type=list_of_floats,
            action="store",
            required=True,
        )

        p.add_argument(
            "--loss_eps",
            action="store",
            dest="loss_eps",
            default=10,
            type=float,
        )

        p.add_argument(
            "--qr",
            action="store",
            dest="qr",
            default=0.5,
            type=float,
        )

        p.add_argument(
            "--resample_ratio",
            action="store",
            dest="resample_ratio",
            default=3,
            type=int,
        )

        p.add_argument(
            "--GF_kernel_size_train",
            action="store",
            dest="GF_kernel_size_train",
            default=29,
            type=int,
        )

        p.add_argument(
            "--GF_kernel_size_inference",
            action="store",
            dest="GF_kernel_size_inference",
            default=29,
            type=int,
        )

        p.add_argument(
            "--hessian_kernel_sigma",
            action="store",
            dest="hessian_kernel_sigma",
            default=1,
            type=float,
        )

        p.add_argument(
            "--sampling_in_MSEloss",
            action="store",
            dest="sampling_in_MSEloss",
            default=2,
            type=int,
        )

        p.add_argument(
            "--isotropic_hessian",
            type=bool_args,
            default="True",
        )

        p.add_argument(
            "--lambda_tv",
            action="store",
            dest="lambda_tv",
            default=1,
            type=float,
        )

        p.add_argument(
            "--lambda_hessian",
            action="store",
            dest="lambda_hessian",
            default=1,
            type=float,
        )

        p.add_argument(
            "--inc",
            action="store",
            dest="inc",
            default=16,
            type=int,
        )

        p.add_argument(
            "--n_epochs",
            action="store",
            dest="n_epochs",
            default=300,
            type=int,
        )

        p.add_argument(
            "--wedge_degree",
            action="store",
            dest="wedge_degree",
            default=29,
            type=float,
        )

        p.add_argument(
            "--n_neighbors",
            action="store",
            dest="n_neighbors",
            default=16,
            type=int,
        )

        p.add_argument(
            "--fast_GF",
            type=bool_args,
            default="False",
        )

        p.add_argument(
            "--require_global_correction",
            type=bool_args,
            default="True",
        )

        p.add_argument(
            "--fusion_GF_kernel_size",
            action="store",
            dest="fusion_GF_kernel_size",
            default=49,
            type=int,
        )

        p.add_argument(
            "--fusion_Gaussian_kernel_size",
            action="store",
            dest="fusion_Gaussian_kernel_size",
            default=49,
            type=int,
        )

        p.add_argument(
            "--X1_path",
            action="store",
            dest="X1_path",
            type=str,
            required=True,
        )

        p.add_argument(
            "--X2_path",
            action="store",
            dest="X2_path",
            default=None,
            type=str,
        )

        p.add_argument(
            "--mask_path",
            action="store",
            dest="mask_path",
            default=None,
            type=str,
        )

        p.add_argument(
            "--boundary",
            action="store",
            dest="boundary",
            default=None,
            type=str,
        )

        p.add_argument(
            "--save_path",
            action="store",
            dest="save_path",
            type=str,
            required=True,
        )

        p.add_argument(
            "--save_path_top_or_left_view",
            action="store",
            dest="save_path_top_or_left_view",
            default=None,
            type=str,
        )

        p.add_argument(
            "--save_path_bottom_or_right_view",
            action="store",
            dest="save_path_bottom_or_right_view",
            default=None,
            type=str,
        )

        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help=argparse.SUPPRESS,
        )
        p.parse_args(namespace=self)


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug

        exe = DeStripe(
            args.loss_eps,
            args.qr,
            args.resample_ratio,
            args.GF_kernel_size_train,
            args.GF_kernel_size_inference,
            args.hessian_kernel_sigma,
            args.sampling_in_MSEloss,
            args.isotropic_hessian,
            args.lambda_tv,
            args.lambda_hessian,
            args.inc,
            args.n_epochs,
            args.wedge_degree,
            args.n_neighbors,
            args.fast_GF,
            args.require_global_correction,
            args.fusion_GF_kernel_size,
            args.fusion_Gaussian_kernel_size,
        )
        out = exe.train(
            args.X1_path,
            args.is_vertical,
            args.angle_offset,
            args.X2_path,
            args.mask_path,
            args.boundary,
        )
        if not isinstance(out, tuple):
            OmeTiffWriter.save(out, args.save_path, dim_order="ZYX")
        else:
            OmeTiffWriter.save(
                out[0],
                args.save_path,
                dim_order="ZYX",
            )
            OmeTiffWriter.save(
                out[1],
                args.save_path_top_or_left_view,
                dim_order="ZYX",
            )
            OmeTiffWriter.save(
                out[2],
                args.save_path_bottom_or_right_view,
                dim_order="ZYX",
            )

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
