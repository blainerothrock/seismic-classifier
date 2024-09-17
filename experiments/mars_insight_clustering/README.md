# Mars Seismic Data
The [Mars InSight Lander](https://mars.nasa.gov/insight/) has been recording seismic data for most of 2019. The data is being released on a 3-month cadence from [SEIS](https://www.seis-insight.eu/en/science/science-summary). There is also a [active catalog](https://www.seis-insight.eu/en/science/seis-products/mqs-catalogs) of events. [Iris](https://www.iris.edu/hq/sis/insight) is also hosting data, but is behind the offical EU site, therefore most data is pulled from SEIS. The paper from Nature Geoscience [__The Sesimicity of Mars__](https://www.nature.com/articles/s41561-020-0539-8) outlines the findings from the first catalog release. More information regarding Insight scientific findings can be found in the Nature issue, [Insight on Mars](https://www.nature.com/collections/iiiifgehfc). 

## Experiment Overview
The current event categorization process at SEIS is done using common seismology techniques by professionals. This experiement proposes using unsupervised machine learning techniques in order to cluster small smaples of the seimic data on mars. The hypothesis is that many of the examples can be clustered to match results found from the SEIS lab. The hops is that some cluster may provide new events or help in the identification of anomalies such as wind, drastic temperature changes, or other instrument interference. Ideally a model that could cluster in such a way could assist in the categorization of data.
* [Deep Clustering Auto-Encoder](unsupervised_clustering/README.md)

## Data
Raw data is pulled directly from the SEIS API using the availability information provided [here](https://www.seis-insight.eu/en/science/seis-data/seis-data-availability). The `XB` Network is used for instruments on Mars and the station `ELYSE` is reserved for scientific data during the mission, post full instrumentation deployment. Channels `BHU`, `BHV` and `BHW` are used to

### Availability
The notebook `mars_event_windowsi.ipynb` (split into scripts with `download_all_data.py` & `prepare_data.py`) will download data directly from [Insight's science portal](https://www.seis-insight.eu) (EU) and split into smaller samples to be fed into models. This is desinged for unsupervised modeling. Model-ready (processed) datasets are hosted on S3 for the frameworks convenience [here](../../seisml/utility/download_data.py).

More notebooks related to retriveing manually labeled events can be found [here](../../playground/mars_insight_seismic). 

### Dataset (`seisml/datasets/mars_insight.py`)
this dataset will apply a transform to each sample read from a directory of `mseed` files. For an example of using this dataset, refer to the [dataset unit test](../../tests/datasets/mars_insight_test.py).

## Events
Manually events have been labeled over this dataset using traditional seismology techniques, outline in [The Sesimicity of Mars](https://www.nature.com/articles/s41561-020-0539-8). We filter these events for exploration and hosted the images with url `https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/mars/event_images/`. To view a particular events with general event find the event Name on [Iris](http://ds.iris.edu/ds/nodes/dmc/tools/mars-events/) and append to the url, examples:

#### Event Name: `S0387a`
* `https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/mars/event_images/S0387a.png`: Bandpass filtered from 1 to 9

![event filtered](https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/mars/event_images/S0387a.png)

* `https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/mars/event_images/S0387a_raw.png`: raw event data

![event raw](https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/mars/event_images/S0387a_raw.png)
 

