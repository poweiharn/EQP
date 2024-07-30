import torch
import torch.nn as nn
import torch.nn.functional as F
from Pooling.get_pooling import get_pooling

sub_tree_identifier = {'00','01','11','10'}
branch_id_list = list(sub_tree_identifier)


class GeneticPool(nn.Module):
    def __init__(self, kernel_size, stride, gene, padding=0):
        super(GeneticPool,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.gene = gene
        #self.index = 0

    def address_pooling(self, gene_node, x, full):
        if gene_node.is_sensor:
            pool = get_pooling({"pooling": gene_node.data})
            return pool(self.kernel_size, self.stride)(x)

        else:
            data_00 = None
            data_01 = None
            data_10 = None
            data_11 = None
            for branch_id in branch_id_list:
                if gene_node.branch[branch_id]:
                    half = full // 2
                    # 00
                    if branch_id == branch_id_list[0]:
                        sub_grid_x = x[:,:,:half,:half]
                        data_00 = self.address_pooling(gene_node.branch[branch_id], sub_grid_x, half)

                    # 01
                    elif branch_id == branch_id_list[1]:
                        sub_grid_x = x[:,:,:half,half:]
                        data_01 = self.address_pooling(gene_node.branch[branch_id], sub_grid_x, half)

                    # 10
                    elif branch_id == branch_id_list[2]:
                        sub_grid_x = x[:,:,half:,:half]
                        data_10 = self.address_pooling(gene_node.branch[branch_id], sub_grid_x, half)

                    # 11
                    elif branch_id == branch_id_list[3]:
                        sub_grid_x = x[:,:,half:,half:]
                        data_11 = self.address_pooling(gene_node.branch[branch_id], sub_grid_x, half)

            tens1 = torch.cat((data_00, data_01), -1)
            tens2 = torch.cat((data_10, data_11), -1)
            x = torch.cat((tens1, tens2), 2)

            return x

    def forward(self, x):
        #gene_node = self.gene[self.index]
        gene_node = self.gene
        _, c, h, w = x.size()
        x = self.address_pooling(gene_node, x, w)
        #self.index = self.index + 1
        return x











    '''
    def forward(self, x):
        gene_node = self.gene[0]
        data = {}
        _, c, h, w = x.size()
        for branch_id in branch_id_list:
            leaf_node = gene_node.branch[branch_id]
            pool = get_pooling({"pooling": leaf_node.data})

            #00
            if branch_id == branch_id_list[0]:
                data[branch_id] = pool(self.kernel_size, self.stride)(x[:,:,:h//2,:w//2])
            #01
            elif branch_id == branch_id_list[1]:
                data[branch_id] = pool(self.kernel_size, self.stride)(x[:,:,:h//2,w//2:])
            #10
            elif branch_id == branch_id_list[2]:
                data[branch_id] = pool(self.kernel_size, self.stride)(x[:,:,h//2:,:w//2])
            #11
            elif branch_id == branch_id_list[3]:
                data[branch_id] = pool(self.kernel_size, self.stride)(x[:,:,h//2:,w//2:])

        tens1 = torch.cat((data["00"], data["01"]), -1)
        tens2 = torch.cat((data["10"], data["11"]), -1)
        x = torch.cat((tens1, tens2), 2)

        return x
        '''

