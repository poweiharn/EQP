import torch
import pickle

VGG_feature_dim = {
    1:[1, 64, 32, 32],
    2:[1, 128, 16, 16],
    3:[1, 256, 8, 8],
    4:[1, 512, 4, 4],
    5:[1, 512, 2, 2]
}

dataset_num = {
    "mnist": 60000+10000,
    "cifar10":50000+10000,
    "cifar100":50000+10000,
    "svhn":73257+26032
}


def count_maxpool(L, leaf_dim, kernel_dim):
    total_ops = ((leaf_dim/kernel_dim))*(kernel_dim - 1)*L[0]*L[1]
    return total_ops

def count_avgpool(L, leaf_dim, kernel_dim):
    total_ops = ((leaf_dim/kernel_dim))*(kernel_dim)*L[0]*L[1]
    return total_ops

def count_stochasticPool(L, leaf_dim, kernel_dim):
    total_ops = ((leaf_dim / kernel_dim)) * (3*kernel_dim) * L[0] * L[1]
    return total_ops

def count_mixedPool(L, leaf_dim, kernel_dim):
    total_ops = count_maxpool(L,leaf_dim, kernel_dim)+count_avgpool(L,leaf_dim, kernel_dim)
    return total_ops

def count_medianPool(L, leaf_dim, kernel_dim):
    total_ops = ((leaf_dim / kernel_dim)) * (kernel_dim*kernel_dim) * L[0] * L[1]
    return total_ops

def count_softPool(L, leaf_dim, kernel_dim):
    total_ops = 2*count_avgpool(L,leaf_dim, kernel_dim)
    return total_ops

sub_tree_identifier = {'00','01','11','10'}
branch_id_list = list(sub_tree_identifier)

def count_op(node, feature_map, full):
    total_op = 0

    if node.is_sensor:
        if node.data == 'M':
            total_op = count_maxpool(feature_map,full*full,2*2)
        elif node.data == 'A':
            total_op = count_avgpool(feature_map,full*full,2*2)
        elif node.data == 'X':
            total_op = count_mixedPool(feature_map, full * full, 2 * 2)
        elif node.data == 'C':
            total_op = count_medianPool(feature_map, full * full, 2 * 2)
        elif node.data == 'S':
            total_op = count_stochasticPool(feature_map, full * full, 2 * 2)
        elif node.data == 'T':
            total_op = count_softPool(feature_map, full * full, 2 * 2)
        return total_op

    else:
        for branch_id in branch_id_list:
            half = full // 2
            total_op = total_op + count_op(node.branch[branch_id], feature_map, half)
        return total_op


def quadtree_op_count(gene):
    total_op = 0
    for i in range(len(gene)):
        total_op = total_op + count_op(gene[i], VGG_feature_dim[i+1], VGG_feature_dim[i+1][3])
    return total_op

def ea_op_count(gene):
    total_op = 0
    for i in range(len(gene)):
        for j in range(len(gene[i])):
            data = gene[i][j]
            if data == 'M':
                total_op = total_op + count_maxpool(VGG_feature_dim[i + 1], 2 * 2, 2 * 2)
            elif data == 'A':
                total_op = total_op + count_avgpool(VGG_feature_dim[i + 1], 2 * 2, 2 * 2)
            elif data == 'X':
                total_op = total_op + count_mixedPool(VGG_feature_dim[i + 1], 2 * 2, 2 * 2)
            elif data == 'C':
                total_op = total_op + count_medianPool(VGG_feature_dim[i + 1], 2 * 2, 2 * 2)
            elif data == 'S':
                total_op = total_op + count_stochasticPool(VGG_feature_dim[i + 1], 2 * 2, 2 * 2)
            elif data == 'T':
                total_op = total_op + count_softPool(VGG_feature_dim[i + 1], 2 * 2, 2 * 2)
    return total_op

def vgg_pooling_count(data):
    total_op = 0
    for i in range(5):
        if data == 'M':
            total_op = total_op + count_maxpool(VGG_feature_dim[i+1], VGG_feature_dim[i+1][3]*VGG_feature_dim[i+1][3], 2 * 2)
        elif data == 'A':
            total_op = total_op + count_avgpool(VGG_feature_dim[i+1], VGG_feature_dim[i+1][3]*VGG_feature_dim[i+1][3], 2 * 2)
        elif data == 'X':
            total_op = total_op + count_mixedPool(VGG_feature_dim[i+1], VGG_feature_dim[i+1][3]*VGG_feature_dim[i+1][3], 2 * 2)
        elif data == 'C':
            total_op = total_op + count_medianPool(VGG_feature_dim[i+1], VGG_feature_dim[i+1][3]*VGG_feature_dim[i+1][3], 2 * 2)
        elif data == 'S':
            total_op = total_op + count_stochasticPool(VGG_feature_dim[i+1], VGG_feature_dim[i+1][3]*VGG_feature_dim[i+1][3], 2 * 2)
        elif data == 'T':
            total_op = total_op + count_softPool(VGG_feature_dim[i+1], VGG_feature_dim[i+1][3]*VGG_feature_dim[i+1][3], 2 * 2)
    return total_op


