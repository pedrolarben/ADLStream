import os
import requests
import json
from datastream_minerva.ADLStream import runADLStream, runARFFProducer, runMultiARFFProducer
from datastream_minerva.models import MODELS

WEBHOOK = os.environ.get('webhook_slack', None)
print(WEBHOOK)

bootstrap_servers = 'localhost:9092'

# arff_folder = './datasets_drift/'
# topics = runMultiARFFProducer(arff_folder, bootstrap_servers)
topics = [
        'streams_ARGWa-F1F4',
        # 'streams_ARGWa-F2F5F8',
        # 'streams_ARGWg-F1F4',
        # 'streams_ARGWg-F2F5F8',
        # 'streams_RBFi-fast',
        # 'streams_RBFi-slow',
        # 'streams_RTGa',
        # 'streams_RTGa3',
        # 'streams_RTGa3D',
        # 'streams_RTGaD',
        # 'streams_RTGg',
        # 'streams_RTGg3',
        # 'streams_RTGg3D',
        # 'streams_RTGgD',
        # 'streams_SEAa-F2F4',
        # 'streams_SEAg-F2F4',
        ]
runARFFProducer('./datasets_drift/ARGWa-F1F4.arff', bootstrap_servers)
i = 0
for t in topics:
    for m in MODELS.keys():
        i += 1
        runADLStream(topic=t,
                     create_model_func=m,
                     two_gpu=True,
                     batch_size=40,
                     num_batches_fed=60,
                     debug=True,
                     output_path='/home/plara/ADLStreamSOCOResults/',
                     from_beginning=True,
                     time_out_ms=10000,
                     bootstrap_servers=bootstrap_servers,
                     clf_name=None)
        if WEBHOOK is not None:
            requests.post(WEBHOOK, json.dumps({'text': 'ADLStream - SOCO {0}/{1}\n\tTopic: {2}\n\tModel: {3}'.format(
                                                str(i), str(len(topics)*len(MODELS)), t, m)}))
