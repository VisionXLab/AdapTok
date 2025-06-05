# ilp
ilp_adap_each_block_t1024_b4 = {
    32: [32, 32, 32, 32],
    64: [142, 40, 36, 36],
    96: [204, 70, 57, 51],
    128: [228, 121, 86, 75],
    160: [239, 171, 123, 105],
    192: [245, 209, 171, 141],
    224: [249, 234, 220, 192],
    256: [256, 256, 256, 256],
}

ilp_adap_each_block_t2048_b4 = {k * 2: [2 *i for i in v] for k, v in ilp_adap_each_block_t1024_b4.items()}


adap_each_block = {
    "ilp_t1024_b4": ilp_adap_each_block_t1024_b4,
    "ilp_t2048_b4": ilp_adap_each_block_t2048_b4
}