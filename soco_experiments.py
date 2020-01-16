import os
import requests
import json
from datastream_minerva.ADLStream import runADLStream, runARFFProducer, runMultiARFFProducer
from datastream_minerva.models import MODELS

WEBHOOK = os.environ.get('webhook_slack')

bootstrap_servers = 'localhost:9092'
arff_folder = './datasets_drift/'
topics = runMultiARFFProducer(arff_folder, bootstrap_servers)

i = 0
for t in topics:
    for m in MODELS.keys():
        i += 1
        runADLStream(topic=t,
                     create_model_func=m,
                     two_gpu=True,
                     batch_size=90,
                     num_batches_fed=60,
                     debug=True,
                     output_path='/home/plara/ADLStreamResults/',
                     from_beginning=True,
                     time_out_ms=10000,
                     bootstrap_servers=bootstrap_servers,
                     clf_name=None)
        requests.post(WEBHOOK, json.dump({'message': 'ADLStream - SOCO {0}/{1}\n\tTopic: {2}\n\tModel: {3}'.format(
            str(i), str(len(topics)*len(MODELS)), t, m)}))
