# 随机生成10个节点的图
import networkx as nx   #导入networkx包
import random 
import matplotlib.pyplot as plt #导入画图工具包
import cpnet
import seaborn as sns
import numpy as np
import os

def rand_edge(vi,vj,p=0.6):		#默认概率p=0.1
    probability =random.random()#生成随机小数
    if(probability>p):			#如果大于p
        G.add_edge(vi,vj)  		#连接vi和vj节点

def CPGraphGenerator(nodes_num, core_num, p = 0.95, verbose=True):  #p controls the sparsity of the generated graphs
    G = nx.Graph()			#建立无向图
    H = nx.path_graph(nodes_num)	#添加节点，10个点的无向图
    G.add_nodes_from(H)		#添加节点

    i = 0
    while (i<nodes_num):
        G.add_edge(i,i)  
        if(i < core_num):
            j=0
            while(j<nodes_num):
                if (j == i):
                    j += 1
                    continue
                if( j < core_num):
                    probability =random.random()#生成随机小数
                    if(probability> (p/random.randint(2, 6))):			#如果大于p
                        G.add_edge(i,j)  		#连接vi和vj节点
                    j +=1
                else:
                    probability =random.random()#生成随机小数
                    if(probability> (p/random.randint(1, 3))):			#如果大于p
                        G.add_edge(i,j)  		#连接vi和vj节点
                    j +=1
            i +=1
        else:
            j=0
            while(j<i):
                probability =random.random()#生成随机小数
                if(probability>p):			#如果大于p
                    G.add_edge(i,j)  		#连接vi和vj节点
                j +=1
            i +=1
    return G

def GraphVis(G):
    algorithm = cpnet.BE()
    algorithm.detect(G)
    c = algorithm.get_pair_id()
    x = algorithm.get_coreness()
    core_nodes_id = [k for k, v in x.items() if v == 1]
    print(core_nodes_id)
    print(len(core_nodes_id))
    #print('adj matrxi\n', nx.to_numpy_array(G))

    # 将生成的图 G 打印出来
    pos = nx.circular_layout(G)
    options = {
        "with_labels": True,
        "font_size": 1,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 50,
        "width": 0.1
    }
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    ax1 = sns.heatmap(nx.to_numpy_array(G))
    plt.show()
    

if __name__ == '__main__':
    cp_graph_path = './cp_graphsV2'
    if (not os.path.exists(cp_graph_path)):
        os.mkdir(cp_graph_path)

    temp = []
    algorithm = cpnet.BE()

    for nodes_num in range(10, 200, 10):
        core_num = nodes_num - 10
        while(core_num > 0):
            for i in range(5000):
                G = CPGraphGenerator(nodes_num, core_num)
                algorithm.detect(G)
                c = algorithm.get_pair_id()
                x = algorithm.get_coreness()
                core_nodes_id = [k for k, v in x.items() if v == 1]
                est_core_num = len(core_nodes_id)
                if( not nx.is_connected(G)):
                    print('not conneteced')
                    continue
                if( G == temp):
                    print('same graph')
                    continue
                if(est_core_num != core_num or core_nodes_id[-1] >= core_num):
                    print('core num issue')
                    continue
                temp = G
                ccs_dic = nx.clustering(G)
                ccs = [ccs_dic[i] for i in ccs_dic.keys()]
                cc = format(nx.average_clustering(G), '.1f')
                L = format(nx.average_shortest_path_length(G), '.1f')
                print(cc)
                print(L)
                print(core_nodes_id)
                nx.write_gexf(G, cp_graph_path+'/node_'+str(nodes_num)+'_core_'+str(core_num)+'_L_'+ str(L)+ '_CC_'+ str(cc) +'.gexf')
            core_num = core_num - 10
                #GraphVis(G)
   
    '''
    G = nx.Graph()			#建立无向图
    H = nx.path_graph(10)	#添加节点，10个点的无向图
    G.add_nodes_from(H)		#添加节点
    
    
    i=0
    while (i<10):
        j=0
        while(j<i):
                rand_edge(i,j)		#调用rand_edge()
                j +=1
        i +=1
    
    # 将生成的图 G 打印出来
    pos = nx.circular_layout(G)
    options = {
        "with_labels": True,
        "font_size": 20,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 2000,
        "width": 2
    }
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    

    #nx.write_gexf(G, 'test.gexf')
    G = nx.read_gexf('test.gexf')

    algorithm = cpnet.BE()
    algorithm.detect(G)
    c = algorithm.get_pair_id()
    x = algorithm.get_coreness()
    core_nodes_id = [k for k, v in x.items() if v == 1]
    print(core_nodes_id)
    print(len(core_nodes_id))
    print('adj matrxi\n', nx.to_numpy_array(G))

    print(nx.is_connected(G))

    # 将生成的图 G 打印出来
    pos = nx.circular_layout(G)
    options = {
        "with_labels": True,
        "font_size": 15,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 1000,
        "width": 2
    }
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    
    print('adj matrxi\n', nx.to_numpy_array(G))
    ax1 = sns.heatmap(nx.to_numpy_array(G))
    plt.show()
    '''
