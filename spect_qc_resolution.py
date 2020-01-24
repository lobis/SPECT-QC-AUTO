# -*- coding: utf-8 -*-

# Author: Luis Antonio Obis Aparicio (luis.antonio.obis@gmail.com)
# Website: https://github.com/lobis/SPECT-QC-AUTO

import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

from pydicom import dcmread

import os
import argparse

def read_camera_file(DCM_file_path, pass_tests=True):
    # verify file exists
    assert os.path.isfile(DCM_file_path), f"unable to access '{os.path.abspath(DCM_file_path)}' please check file path is correct"
    ds = dcmread(DCM_file_path)
    if pass_tests:
        # verify file has the correct format
        # it should contain 2 512x512 images
        assert len(ds.pixel_array) > 1, "DCM file does not contain image data"
        requiered_resolution = (512, 512)
        for image in ds.pixel_array:
            assert image.shape == requiered_resolution, f"image resolution is {image.shape} instead of {requiered_resolution}"
    
    return ds

def distribution_to_points(data):
    X = []
    Y = []
    for i in range(data.shape[0]): # Y
        for j in range(data.shape[1]): # X
            value = data[i,j]
            for k in range(value):
                X.append(j)
                Y.append(i)
    assert (len(X) == len(Y) == sum(sum(data)))

    points = []
    for x,y in zip(X,Y):
        points.append([x,y])
    points = np.array(points)
    
    return points

def compute_kmeans_centroids(pixel_array, n_clusters=4):
    points = distribution_to_points(pixel_array)
    
    kmeans = KMeans(n_clusters=4)
    kmeans = kmeans.fit(points)

    centroids = kmeans.cluster_centers_
    # sort them
    mid_x = centroids[:,0].mean()
    mid_y = centroids[:,1].mean()
    def f(e):
        angle = np.arctan2(e[1] - mid_y, e[0] - mid_x)
        return angle
    centroids = [centroid for centroid in centroids]
    centroids.sort(key=f, reverse=True)
    centroids = np.array(centroids)

    return centroids
    
def get_isolated_peaks(points, centroids, radius):
    # radius in px
    reduction_factor = 0.9
    peak_points = [[] for i in range(len(centroids))]
    for point in points:
        n_centroids = 0 # this has to be equal to one meaning only one centroid is close enough so overlapping is prevented
        for i, centroid in enumerate(centroids):
            radial_dist = np.linalg.norm(point - centroid)
            if radial_dist <= radius:
                peak_points[i].append([point[0], point[1]])
                n_centroids += 1
        if n_centroids > 1:
            # radius is too big, abort
            print(f"DEBUG: radius {radius} is too big, trying with {radius*reduction_factor}")
            break
    if n_centroids > 1:
        return get_isolated_peaks(points, centroids, radius=radius*reduction_factor)
    else:
        for i in range(len(peak_points)):
            peak_points[i] = np.array(peak_points[i])
        return peak_points
    
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
def fit_gauss(x, y):
    # initial guess of coefficients is important
    mu = np.mean(x)
    sigma = 1
    A = max(y)
    p0 = [A, mu, sigma]
    coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
    
    #x_fit = np.linspace(min(x),max(x),200)
    #hist_fit = gauss(x_fit, *coeff)
    
    return coeff

def plot_graphs(peak_hists, camera_pixel_data, save=False, camera_id="not-specified"):
    # first we get the maximum to have the same y axis limits in all plots
    ylim_upper = 0
    for i in range(len(peak_hists)):
        data = peak_hists[i]
        ylim_upper = max(max(max(data.sum(axis=0)), max(data.sum(axis=1))), ylim_upper)
    ylim_upper = ylim_upper - ylim_upper % 500 + 500

    output_files = []
    for i in range(len(peak_hists)):
        centroid = centroids[i]
        data = peak_hists[i]

        x = np.linspace(centroid[0] - width/2, centroid[0] + width/2, width)
        data_x = data.sum(axis=0)
        x_coeff = fit_gauss(x, data_x)
        x_fit = np.linspace(min(x), max(x), 200)
        x_FWHM = 2*np.sqrt(2*np.log(2))*x_coeff[2]

        y = np.linspace(centroid[1] - width/2, centroid[1] + width/2, width)
        data_y = data.sum(axis=1)
        y_coeff = fit_gauss(y, data_y)
        y_fit = np.linspace(min(y), max(y), 200)
        y_FWHM = 2*np.sqrt(2*np.log(2))*y_coeff[2]

        fig, ax = plt.subplots(1,3, figsize=(10, 5), constrained_layout=True)

        ax[0].bar(x, data_x, color="b", alpha=0.75, label=f"FWHM = {x_FWHM:1.2f} px")
        ax[0].set_xlabel("X (px)")
        ax[0].plot(x_fit, gauss(x_fit, *x_coeff), color="black")

        ax[1].bar(y, data_y, color="r", alpha=0.75, label=f"FWHM = {y_FWHM:1.2f} px")
        ax[1].set_xlabel("Y (px)")
        ax[1].plot(y_fit, gauss(y_fit, *y_coeff), color="black")

        ax[1].set_yticks([])

        ax[0].set_ylim([0, ylim_upper])
        ax[1].set_ylim([0, ylim_upper])

        #ax[0].legend()
        #ax[1].legend()

        ax[2].imshow(camera_pixel_data, alpha=0.5, cmap='gray', vmin=0, vmax=255)
        circle = matplotlib.patches.Ellipse((x_coeff[1],y_coeff[1]), x_coeff[2]*10, x_coeff[2]*10, fill=False, color="black")
        ax[2].add_artist(circle)
        ax[2].set_xlabel("X (px)")
        ax[2].set_ylabel("Y (px)")
        ax[2].set_xlim([0, 512])
        ax[2].set_ylim([0, 512])
        #ax[2].legend()
        
        fig.suptitle(f"camera {camera_id} - peak {i+1}\nX FWHM = {x_FWHM:1.2f} px\nY FWHM = {y_FWHM:1.2f} px\nIntensity = {int(sum(data_x)):d}", fontsize=14)
        #fig.tight_layout()
        print(f"camera {camera_id} - peak {i+1} FWHM_X = {x_FWHM:1.2f} FWHM_Y = {y_FWHM:1.2f}")
        if save:
            name = os.path.abspath(f"output/{dicom_file}_camera_{camera_id}_peak_{i+1}.pdf")
            try:
                os.mkdir(os.path.abspath("output"))
            except FileExistsError:
                pass
            plt.savefig(name)
            output_files.append(name)
        #plt.show()
    return output_files

def merge_pdf(pdfs):
    from PyPDF2 import PdfFileMerger
    merger = PdfFileMerger()

    for pdf in sum(pdfs, []):
        merger.append(pdf)

    output_filename = f"{dicom_file}_analysis_result.pdf"
    merger.write(output_filename)
    print(f"saving report to {output_filename}")
    merger.close()


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", help="DICOM file to analyze", action="extend", nargs="+", type=str)
args = parser.parse_args()

for filename in args.files:
    filename = filename.replace("\\", "/")
    dicom_file = filename.split("/")[-1]
    print(f"processing {os.path.abspath(filename)}")
    ds = read_camera_file(filename, pass_tests=False)
    output_files = []
    
    for i, camera_pixel_data in enumerate(ds.pixel_array):
        points = distribution_to_points(camera_pixel_data)
        centroids = compute_kmeans_centroids(camera_pixel_data)
    
        width = 60
        peak_points = get_isolated_peaks(points, centroids, radius=width/2)
        peak_hists = {i: 
                     camera_pixel_data[int(centroids[i][1] - width/2) : int(centroids[i][1] + width/2) , int(centroids[i][0] - width/2) : int(centroids[i][0] + width/2)]
                     for i in range(len(peak_points))
                    }
    
        ###
        output_files.append(plot_graphs(peak_hists, camera_pixel_data, save=True, camera_id=i+1))
        
    merge_pdf(output_files)