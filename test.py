from ADLStream.ADLStream import runADLStream, runARFFProducer

bootstrap_servers = 'localhost:9092'
arff_file = './datasets_arff/breast.arff'
topic = runARFFProducer(arff_file, bootstrap_servers)

runADLStream(topic, 
             output_path='ADLStream_example_output/', 
             bootstrap_servers=bootstrap_servers, 
             two_gpu=False,
             debug=True, create_model_func='lstm', 
             clf_name='ADLStream_test')
