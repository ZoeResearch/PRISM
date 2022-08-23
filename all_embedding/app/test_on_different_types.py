import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect', '-d')
    parser.add_argument('--embed', '-e')
    args = parser.parse_args()

    violation_doc_path = "./data/src_top_20_with_time_newest_with_warning_context_for_warning_type"
    save_base = "./score/" + args.embed + "_" + args.detect + "/"
    pre_trained_embed_base = "../pre_train_def/spotbugs/"

    batch_size_range = [32]
    epochs_d_range = [40, 50]
    # epochs_d_range = [1]
    lstm_unit_range = [32, 64]  # fixed
    optimizer_range = ["Adam"]  # fixed
    layer_range = [6, 8, 10]  # fixed
    drop_out_range = [0.5]
    learning_rate_range = [0.0003, 0.001]
    gru_unit_range = [256, 512]
    dense_unit_range = [32, 64, 128]
    pool_size_range = [15, 20, 30]
    kernel_size_range = [5, 10, 15]
    detect_arg = dict(batch_size_range=batch_size_range, epochs_d_range=epochs_d_range,
                      lstm_unit_range=lstm_unit_range, optimizer_range=optimizer_range,
                      layer_range=layer_range, drop_out_range=drop_out_range,
                      learning_rate_range=learning_rate_range, gru_unit_range=gru_unit_range,
                      dense_unit_range=dense_unit_range, pool_size_range=pool_size_range,
                      kernel_size_range=kernel_size_range)