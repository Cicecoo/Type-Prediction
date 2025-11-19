(naturalcc) zhaojunzhang@dlserver6-Super-Server:~/workspace/type_pred/naturalcc$ python convert_typilus_to_transformer.py \
>   --typilus-dir /mnt/data1/zhaojunzhang/typilus-data/typilus \
>   --output-dir /mnt/data1/zhaojunzhang/typilus-data/transformer

============================================================
Converting train split
============================================================
Converting train split from attributes...
Loading vocabulary...
Loaded 9999 tokens from vocabulary
Reading from /mnt/data1/zhaojunzhang/typilus-data/typilus/attributes/train.token-sequence, /mnt/data1/zhaojunzhang/typilus-data/typilus/attributes/train.nodes, /mnt/data1/zhaojunzhang/typilus-data/typilus/attributes/train.supernodes
Writing to /mnt/data1/zhaojunzhang/typilus-data/transformer/train.code, /mnt/data1/zhaojunzhang/typilus-data/transformer/train.type
Processing train: 9640it [00:05, 3374.03it/s]Error processing line 9684: object of type 'NoneType' has no len()
Error processing line 10006: object of type 'NoneType' has no len()
Processing train: 10626it [00:05, 4005.40it/s]Error processing line 10829: object of type 'NoneType' has no len()
Processing train: 20640it [00:08, 4426.21it/s]Error processing line 20842: object of type 'NoneType' has no len()
Error processing line 20855: object of type 'NoneType' has no len()
Processing train: 22921it [00:08, 5181.44it/s]Error processing line 23037: object of type 'NoneType' has no len()
Processing train: 31498it [00:10, 2808.63it/s]Error processing line 31492: object of type 'NoneType' has no len()
Processing train: 37580it [00:12, 3923.46it/s]Error processing line 37597: object of type 'NoneType' has no len()
Processing train: 37982it [00:12, 3908.57it/s]Error processing line 38015: object of type 'NoneType' has no len()
Error processing line 38024: object of type 'NoneType' has no len()
Error processing line 38057: object of type 'NoneType' has no len()
Error processing line 38099: object of type 'NoneType' has no len()


Traceback (most recent call last):
  File "run/type_prediction/transformer/train.py", line 339, in <module>
    cli_main()
  File "run/type_prediction/transformer/train.py", line 332, in cli_main
    single_main(args)
  File "run/type_prediction/transformer/train.py", line 211, in single_main
    task = tasks.setup_task(args)
  File "/home/zhaojunzhang/workspace/type_pred/naturalcc/ncc/tasks/__init__.py", line 15, in setup_task
    return TASK_REGISTRY[args['common']['task']].setup_task(args, **kwargs)
  File 


  