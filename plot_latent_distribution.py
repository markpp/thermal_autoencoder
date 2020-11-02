#!/usr/bin/env python
import numpy as np
import pca_reducer
import matplotlib.pyplot as plt
import pandas as pd

import argparse

import os
import json
import random

import seaborn as sns

import datashader as ds
import datashader.utils as utils
import datashader.transfer_functions as tf
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral6
from io import BytesIO
from PIL import Image
import base64

class interactive_scatter_plot:
    def __init__(self):
        output_file("output/bokeh_hover_plot.html")

    # Plotting embeddings
    def encode_img_png(self,data):
        #data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = data.reshape(64,64)
        print(data.shape)
        image = Image.fromarray(data, mode='L')
        buffer = BytesIO()
        image.save(buffer, format='png')
        for_encoding = buffer.getvalue()
        return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

    def plot_emb_bokeh(self,points,names,labels,target_names,images):
        plot_df = pd.DataFrame(points, columns=('x', 'y'))
        plot_df['sample'] = [str(x) for x in names]
        plot_df['image'] = list(map(self.encode_img_png, images))
        plot_df['cluster'] = [x for x in labels]

        datasource = ColumnDataSource(plot_df)
        color_mapping = CategoricalColorMapper(factors=target_names,
                                               #palette=Spectral6)
                                               palette=['#00ff00', '#0000ff'])

        plot_figure = figure(
            title='PCA projection of the X dataset',
            plot_width=1400,
            plot_height=1000,
            tools=('pan, wheel_zoom, reset')
        )

        plot_figure.add_tools(HoverTool(tooltips="""
        <div>
            <div>
                <img src='@image' height="512" width="512" style='float: left; margin: 5px 5px 5px 5px'/>
            </div>
            <div>
                <span style='font-size: 16px; color: #224499'>Sample:</span>
                <span style='font-size: 18px'>@sample</span>
            </div>
        </div>
        """))

        plot_figure.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='cluster', transform=color_mapping),
            line_alpha=0.8,
            fill_alpha=0.5,
            size=12
        )
        show(plot_figure)

from pyntcloud import PyntCloud

def save_point_cloud(points, colors=None):
    if len(colors) > 0:
        print(np.concatenate((points,colors), axis=1).shape)
        df = pd.DataFrame(np.concatenate((points,colors), axis=1), columns=['x', 'y', 'z', 'red', 'green', 'blue'])
    else:
        df = pd.DataFrame(points, columns=['x', 'y', 'z'])

    point_cloud = PyntCloud(df)
    point_cloud.to_file("output/pca_3d.ply",as_text=True)

if __name__ == '__main__':
    """
    Command:
        python points2yolo.py -l -s
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-la", "--latent", type=str,
                    default='output/latent.npy', help="Path to .npy with features")
    ap.add_argument("-lo", "--losses", type=str,
                    default='output/losses.npy', help="Path to .npy with losses")
    ap.add_argument("-in", "--inputs", type=str,
                    default='output/inputs.npy', help="Path to .npy with images")
    # add val features

    # use colors for labels and symbol for dataset
    args = vars(ap.parse_args())

    # load full images
    features = np.load(args["latent"], allow_pickle=True)
    print(features.shape)
    losses = np.load(args["losses"], allow_pickle=True)
    losses = np.stack([str(x) for x in losses])
    print(losses.shape)
    print(losses[0])
    inputs = np.load(args["inputs"], allow_pickle=True)

    colors = np.stack([[255,0,0] for x in features])



    # pca dim reduction
    pca_red = pca_reducer.pca_reducer()
    pca_red.create_transform(features)
    pca_red.save_transform('output/pca_transform.pkl')
    pca_components = pca_red.transform_enc(features)
    np.save("output/pca_components.npy", pca_components)
    #pca_red.plot_main_components(pca_points[:testset_end,:],pca_points[testset_end:,:])
    #pca_red.plot_all_components(pca_components[:100,:],pca_components[100:,:])
    pca_red.info()

    save_point_cloud(pca_components[:,:3], colors)

    interactive_plotter = interactive_scatter_plot()

    interactive_plotter.plot_emb_bokeh(pca_components[:,:2],losses,losses,losses,inputs)


#sns.distplot(anno_df['z'], rug=True, hist=True)
#sns.jointplot(x='x', y='emb_0', data=df)
#sns.pairplot(df[['x','y','z','emb_0','emb_1']])
#plt.show()
