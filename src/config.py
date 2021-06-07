class opt:
    dataset_name = "pix2pix"
    batch_size = 3
    num_workers = 4
    n_epochs = 200
    channels = 3
    img_height = 224
    img_width = 224
    checkpoint_interval = 5
    n_residual_blocks = 9
    lambda_cyc = 10
    lambda_id = 5
    b1 = 0.5
    b2 = 0.999
    lr = 0.0002
    epoch = 0
    decay_epoch = 100


