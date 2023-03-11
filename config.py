""""
Command Lines
"""
import argparse
parser = argparse.ArgumentParser("Command lines for ONN tensorflow")

# optical parameters
parser.add_argument('--ds', type = int, default = 1, help="Downsample factor of the dimensions")
parser.add_argument('--ps', type = float, default = 25.14e-3, help="Plane spacing(m)")
parser.add_argument('--lamb', type = float, default = 1565e-9, help="Wavelength(m)")
parser.add_argument('--pix', type = float, default = 8e-6, help="Pixel dimension(m)")
parser.add_argument('--sz', type = int, default=256, help="Iuput image size")

# training hyperparameters
parser.add_argument('--ep', type = int, default = 10, help="Number of epochs")
parser.add_argument('--lr', type = float, default = 1e-2, help="Learning rate")
parser.add_argument('--bt', type = int, default = 20, help="Batch size")
parser.add_argument('--mo_name', type = str, default = "Test_ONN", help="Name of the output model")

args = parser.parse_args()

