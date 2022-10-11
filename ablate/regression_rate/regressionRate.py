import argparse
import math
import pathlib
import matplotlib.pyplot as plt

from support.curveFileReader import parse_curve_file
from support.supportPaths import expand_path

# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots the regression rate for ablate slab burner simulations')
    parser.add_argument('--file', dest='curve_file', type=pathlib.Path, required=True,
                        help='The path to the curve file(s) containing the structured data.  A wild card can be used '
                             'to supply more than one file.')
    parser.add_argument('--density', dest='density', type=float, required=False, default=900,
                        help='The density to compute the regression rate')
    parser.add_argument('--timePlot', dest='timePlot', type=bool, required=False, default=False,
                        help='If a plot is generated for each time in the time series')
    args = parser.parse_args()

    # create an empty data frame
    regression_rates_rms = None

    # Set some values
    xAxis = 'x'
    regressionMassFluxName = 'slab boundary_monitor_regressionMassFlux'
    regressionRateName = 'regression_rate mm/s'
    regressionRateRmsName = 'regression_rate_rms'

    # expand any wild cards in the path name
    curve_paths = expand_path(args.curve_file)

    # extract the density
    density = args.density
    count = 0

    # for each cure_path
    for curve_path in curve_paths:
        # create component markdown
        regression_rates = parse_curve_file(curve_path)

        # compute the regression rate
        regression_rates[regressionRateName] = regression_rates.apply(
            lambda row: row[regressionMassFluxName] / density * 1000, axis=1)

        regression_rates_sorted = regression_rates.sort_values(by=[xAxis])
        if args.timePlot:
            regression_rates_sorted.plot.line(x='x', y=regressionRateName,
                                              title=f'Regression Rate {regression_rates_sorted["time"][0]}')
            plt.show()

        # merge them together
        count += 1
        if regression_rates_rms is not None:
            # else we need to merge
            regression_rates_rms[regressionRateName] = regression_rates[regressionRateName]
            regression_rates_rms[regressionRateRmsName] = regression_rates_rms.apply(
                lambda row: row[regressionRateName] ** 2 + row[regressionRateRmsName], axis=1)

        else:
            regression_rates_rms = regression_rates.filter([xAxis], axis=1)
            regression_rates_rms[regressionRateRmsName] = regression_rates.apply(
                lambda row: row[regressionRateName] ** 2, axis=1)

    # finish the rms
    regression_rates_rms[regressionRateRmsName] = regression_rates_rms.apply(
        lambda row: math.sqrt(row[regressionRateRmsName] / count), axis=1)
    regression_rates_rms_sorted = regression_rates_rms.sort_values(by=[xAxis])
    regression_rates_rms_sorted.plot.line(x='x', y=regressionRateRmsName, title=f'Regression Rate RMS')
    plt.show()
