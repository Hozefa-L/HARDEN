import networkx as nx
import matplotlib.pyplot as plt
import pylab
import dgl
from sc_opcodes import Opcode_integers 
from outputlabel_final import graph_labels
from dgl.data import DGLDataset
import os
os.environ["DGLBACKEND"] = "pytorch"
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn import GATConv
import random
import json
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
import time

class NodeReader:
    def __init__(self, dot_path):
        self.dot_path = dot_path
        self.Graph = nx.DiGraph()

    def read_networkx_graph(self):
        self.Graph = nx.drawing.nx_pydot.read_dot(self.dot_path)
        self.Node_label = nx.get_node_attributes(self.Graph, 'label')
        self.Fillcolor = nx.get_node_attributes(self.Graph,'fillcolor')
        self.Shape = nx.get_node_attributes(self.Graph, 'shape')
        self.Color = nx.get_node_attributes(self.Graph,'color')
        
        # Create a dictionary to store attributes
        self.attributes_dict = {}

        # Loop through each attribute and apply formatting
        for node, attribute in self.Node_label.items():
           attributes = attribute.split("\\l")
           formatted_string = "\n".join(attributes)
           self.attributes_dict[node] = formatted_string   
        return self.Graph, self.attributes_dict, self.Fillcolor , self.Shape, self.Color
    
    def label_opcodes(self, opcodes_attributes):
        for node, opcodes in opcodes_attributes.items(): # Loop through each node and its opcode string
            opcode_lines = opcodes.strip().split('\n') # Split the opcode string into lines
            degree = 0 # Initialize degrees as node keys
            
            modified_opcodes = [] # For each value string only keep opcode removing extra info
            for line in opcode_lines:
                opcode = line.split(": ", 1)[-1].strip()  # remove node degrees
                if opcode and opcode != "\"":
                    modified_opcodes.append(f"{degree}: {opcode}")
                    degree += 1
            modified_opcodes_str = modified_opcodes
            opcodes_attributes[node] = modified_opcodes_str
            
        opcode_only = {}
        for nodek, opcode in opcodes_attributes.items():
            opcode_only[nodek] = [instruction.split(': ', 1)[1].split(' ', 1)[0] for instruction in opcode] # Only save opcodes for each collected node attributes
        return opcode_only
    
    def draw_networkx_graph(self, Node_Graph, Node_Attributes):
        pos = nx.kamada_kawai_layout(Node_Graph)
        nx.draw(
            Node_Graph,
            pos,
            labels=Node_Attributes,
            with_labels=True,
            node_size=2000,
            node_color="skyblue",
            font_size=4,
            font_color="black",
            arrowsize=20,
        )
        plt.show()

    def collect_function_nodes(self, nodelabels, nodefillcolor, nodeshape, nodecolor):

        self.label = nodelabels
        self.fillcolor = nodefillcolor
        self.shape = nodeshape
        self.color = nodecolor
    
        function_labels = {}
        for node, value in nodelabels.items():
            if nodefillcolor.get(node) is None and nodeshape.get(node) is None and nodecolor.get(node) is None:
                function_labels[node] = value
        return function_labels
       
    def create_feature_nodes(self, Node_Attributes, fun_node_labels):
        New_Node_Attributes = {}
        major_nodes_list = []
        minor_nodes_list = []
        major_nodes_attributes = {}
        minor_nodes_attributes = {}
        minor_mod_nodes_attributes = {}

        for nodekey, opcodeval in Node_Attributes.items():
            New_Node_Attributes[nodekey] = [instruction.split(': ', 1)[1].split(' ', 1)[0] for instruction in opcodeval]
        

        for node, opcodes in New_Node_Attributes.items():
            opcode_count = len(opcodes)
            if opcode_count > 5:
                major_nodes_list.append(node)
                major_nodes_attributes[node] = opcodes
            else:
                minor_nodes_list.append(node)
                minor_nodes_attributes[node] = opcodes
                if opcode_count > 1:
                    minor_mod_nodes_attributes[node] = [opcodes[0], opcodes[-1]]  # Save first and last opcodes
                else:
                    minor_mod_nodes_attributes[node] = [opcodes[0]]

        # Make sure major nodes are also included in minor_nodes_attributes
        for nod in major_nodes_list:
            minor_mod_nodes_attributes[nod] = New_Node_Attributes[nod]

        # Create major_nodes_function_attributes by removing common nodes from function_labels and major_nodes_attributes
        major_nodes_function_attributes = {}
        for node in major_nodes_list:
            if node in New_Node_Attributes and node in fun_node_labels:
                major_nodes_function_attributes[node] = New_Node_Attributes[node]
                del major_nodes_attributes[node]

        return major_nodes_attributes, minor_mod_nodes_attributes, major_nodes_function_attributes
    
    def draw_networkx_subgraph(self, Node_graph, major_node_attrs, minor_node_attrs, major_nodes_function_attributes):
        major_subgraph = Node_graph.subgraph(major_node_attrs)
        minor_subgraph = Node_graph.subgraph(minor_node_attrs)
        major_function_subgraph = Node_graph.subgraph(major_nodes_function_attributes)
        final_graph = nx.compose_all([major_subgraph, minor_subgraph, major_function_subgraph])

        pos = nx.kamada_kawai_layout(Node_graph)

        plt.figure(figsize=(10, 6))


        # Draw merged subgraph (nodes with different colors)
        nx.draw(
            final_graph,
            pos,
            labels={**major_node_attrs, **minor_node_attrs, **major_nodes_function_attributes},
            node_size=[5000 if node in major_function_subgraph else 3000 if node in major_subgraph else 500 for node in final_graph],
            node_color=['orange' if node in major_function_subgraph else 'red' if node in major_subgraph else 'green' for node in final_graph],
            font_size = 5,
            font_color='black',
            arrowsize=20,
            with_labels=True,
        )
        
        plt.show()
    
 
class OpcodeEmbed:
    def __init__(self, opcode_dict, integer_opcode_dict):
        self.opcode_dict = opcode_dict
        self.integer_opcode_dict = integer_opcode_dict
    
    def map_opcode_with_keys(self):
        mapped_dict = {}

        for key, values in self.opcode_dict.items():
            mapped_values = [self.integer_opcode_dict[value] for value in values if value != 'EXIT']
            mapped_dict[key] = mapped_values
        return mapped_dict
    
    def onehotvector(self, map_opcodes):
        node_vector = {}

        for nodes, oploc in map_opcodes.items():
            vector = [0] * 256         
            for loc in oploc:
                try:
                    index = int(loc)
                    if 0 <= index < 256:
                        vector[index] = 1
                except ValueError:
                    print(f"Skipping non-integer value: {oploc}")
            node_vector[nodes] = vector 
        return node_vector 

    def vectormap(self, node_vec, Node_Graph):
        # Combine node attributes with the NetworkX graph
        for node_id, attrs in node_vec.items():
            if node_id in Node_Graph.nodes:
                Node_Graph.nodes[node_id]['emb'] = attrs
        return Node_Graph
    
    def networkxtodgl(self, Nodegraph):
        original_node_ids = list(Nodegraph.nodes) #Create a list of all nodes
        node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(original_node_ids)} # Remap original to new nodes starting from 0
        nx.relabel_nodes(Nodegraph, node_id_mapping, copy=False) # Assign new node keys to netwworkx
        dglgraph = dgl.from_networkx(Nodegraph, node_attrs={'emb'}) # Convert Networkx to DGL graph
        #dglgraph = dgl.to_homogeneous(dglgraph)
        return dglgraph
    
    def drawdglwithnetworkx(self, DGLgraph):
        nxgraph = DGLgraph.to_networkx(node_attrs=['emb']) # Convert back the DGL graph to a NetworkX graph
        pos = nx.kamada_kawai_layout(nxgraph) # Plot the NetworkX graph
        nx.draw(nxgraph, pos, with_labels=True, node_size=2000, node_color = 'green', font_size=10)
        pylab.show()


class SCGraphDataset(DGLDataset):
    def __init__(self, dot_folder, dot_labels):
        self.dot_folder = dot_folder
        self.dot_labels = dot_labels
        super().__init__(name="SCOpcodeGraph")
        
    def process(self):
        dot_files = sorted([file for file in os.listdir(self.dot_folder) if file.endswith(".dot")])
        self.dgl_graphs = []
        self.dgl_labels = []

        for dfile in dot_files:
            dot_file_path = os.path.join(self.dot_folder, dfile)
            dgl_graph = self.dot2dgl(dot_file_path)
            dgl_graph = dgl.add_self_loop(dgl_graph) # added self loops for nodes as model throws error without it
            self.dgl_graphs.append(dgl_graph)
            label= self.get_label_for_graph(dfile)
            self.dgl_labels.append(label)
        self.dgl_labels = th.LongTensor(self.dgl_labels)
        
    # process dot file from ethersolve using NodeReader and OpcodeEmbed to generate dgl graph with node labels   
    def dot2dgl(self, dot_file_path):
        flow_graph_file = NodeReader(dot_file_path)
        node_graph, node_labels, node_fillcolor, node_shape, node_color = flow_graph_file.read_networkx_graph()
        node_to_remove = "\\n"
        if node_to_remove in node_graph.nodes:
            node_graph.remove_node(node_to_remove)
        Opcode_graph = flow_graph_file.label_opcodes(node_labels)
        Opcode_embed = OpcodeEmbed(Opcode_graph, Opcode_integers)
        Op_map_dict = Opcode_embed.map_opcode_with_keys()
        Node_emb_dict = Opcode_embed.onehotvector(Op_map_dict)
        node_graph = Opcode_embed.vectormap(Node_emb_dict, node_graph)
        dgl_graph = Opcode_embed.networkxtodgl(node_graph)
        return dgl_graph
    
    def get_label_for_graph(self, graph_name):
        graph_name_new = os.path.splitext(graph_name)[0]
        for label_info in self.dot_labels:
            if label_info["graph"] == graph_name_new:
                return int(label_info["label"])
        return -1  # Label not found
    
    def __getitem__(self,i):
        return self.dgl_graphs[i] , self.dgl_labels[i]
    
    def __len__(self):
        return len(self.dgl_graphs)
        

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        # self.allow_zero_in_degree = True #does not work with zero in-degrees
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["emb"] = h
        return dgl.mean_nodes(g, "emb")


class ProcessModel:
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, num_epochs=50):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        # self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.train_losses = []
        self.train_accuracies = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1s = []
        self.test_accuracy = 0.0
        self.test_precision = 0.0
        self.test_recall = 0.0
        self.test_f1 = 0.0


    def train(self):
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            y_true_train = []  # For plotting f1, precision and recall of train dataset
            y_pred_train = []  # For plotting f1, precision and recall of train dataset
            start_time = time.time()

            for batched_graph, labels in self.train_dataloader:
                pred_train = self.model(batched_graph, batched_graph.ndata["emb"].float())
                loss_train = F.cross_entropy(pred_train, labels) # For GCN Model
                # loss = F.binary_cross_entropy_with_logits(pred, labels.float())
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

                total_loss += loss_train.item()
                total_correct += (pred_train.argmax(1) == labels).sum().item()
                total_samples += len(labels)

                pred_labels_train = pred_train.argmax(1).cpu().numpy()
                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(pred_labels_train)

            end_time = time.time()
            average_loss = total_loss / len(self.train_dataloader)
            training_accuracy = total_correct / total_samples

            self.train_losses.append(average_loss)
            self.train_accuracies.append(training_accuracy)
            
            self.train_precision = precision_score(y_true_train, y_pred_train, average='binary')
            self.train_precisions.append(self.train_precision)
            self.train_recall = recall_score(y_true_train, y_pred_train, average='binary')
            self.train_recalls.append(self.train_recall)
            self.train_f1 = f1_score(y_true_train, y_pred_train, average='binary')
            self.train_f1s.append(self.train_f1)
            # test_correct_val = 0
            # num_tests_val = 0

            # for batched_graph, labels in self.val_dataloader:
            #     pred_val = model(batched_graph, batched_graph.ndata["emb"].float())
            #     test_correct_val += (pred_val.argmax(1) == labels).sum().item()
            #     num_tests_val += len(labels)

            # val_accuracy = test_correct_val / num_tests_val

            print(f"Iteration num: [{epoch + 1}/{self.num_epochs}]",
                  f"Loss during Training: {average_loss:.4f}",
                  f"Accuracy while Training: {training_accuracy:.4f}",
                  f"Training precision: {self.train_precision:.4f}",
                  f"Training recall: {self.train_recall:.4f}",
                  f"Training f1: {self.train_f1:.4f}",
                  f"Time Training: {(end_time-start_time)*10**3:.4f}ms")

    def test(self):
        test_correct = 0
        num_tests = 0
        y_true_test = []
        y_pred_test = []
        self.model.eval() # For test on real dataset, predict model without gradient
        start_time_test = time.time()

        with th.no_grad(): 
            for batched_graph, labels in self.test_dataloader:
                pred_test = self.model(batched_graph, batched_graph.ndata["emb"].float())
                test_correct += (pred_test.argmax(1) == labels).sum().item()
                num_tests += len(labels)

                pred_labels_test = pred_test.argmax(1).cpu().numpy()
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(pred_labels_test)

        self.test_accuracy = test_correct / num_tests
        self.model.train()
        end_time_test = time.time()

        self.test_precision = precision_score(y_true_test, y_pred_test, average='binary')
        self.test_recall = recall_score(y_true_test, y_pred_test, average='binary')
        self.test_f1 = f1_score(y_true_test, y_pred_test, average='binary')
        print(f"Test Accuracy: {self.test_accuracy:.4f}",
              f"Test Precision: {self.test_precision:.4f}",
              f"Test Recall: {self.test_recall:.4f}",
              f"Test F1-Score: {self.test_f1:.4f}",
              f"Test time: {(end_time_test-start_time_test)*10**3:.4f}ms")

    def plot(self):
        plt.figure(figsize=((800/192),(800/192)))
        #plt.subplot(121)
        #plt.plot(self.train_losses,'r-o',label='Loss in Reentrancy Detection (GCN Train)')
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.legend()
        
        plt.subplot(311)
        plt.plot(self.train_losses, 'r-o',label='Training Loss in Reentrancy Detection')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid()
        plt.subplot(312)
        plt.plot(self.train_accuracies, 'g-o',label='Reentrancy Detection Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Training Accuracy')
        plt.legend()
        plt.grid()
        #plt.subplot(513)
        #plt.plot(self.train_precisions, 'm-.',label='Reentrancy Detection Precision (GCN Train)')
        #plt.xlabel('Epoch')
        #plt.ylabel('Training Precision')
        #plt.legend()
        #plt.subplot(514)
        #plt.plot(self.train_recalls, 'c-',label='Reentrancy Detection Recall (GCN Train)')
        #plt.xlabel('Epoch')
        #plt.ylabel('Training Recall')
        #plt.legend()
        plt.subplot(313)
        plt.plot(self.train_f1s, 'b-o',label='Reentrancy Detection F1-score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.legend()
        plt.grid()
        plt.show()

    def run(self):
        self.train()
        self.test()
        self.plot()


if __name__ == "__main__":
    
    # Create an instance of CFG dataset
    dot_folder = "Reentrancy_dataset/cfg-reentrancy"
    graph_dataset = SCGraphDataset(dot_folder, graph_labels) # Total equal split of  342 reentrancy SC dot files
    # Process the dot files and merge the graphs
    graph_dataset.process()

    # Access the merged DGL graph
    # dgl_graph = graph_dataset.dgl_graphs # Load graphs
    # dgl_labels = graph_dataset.dgl_labels # Load labels for each graph
    # print(dgl_graph)
    # print(dgl_labels)
    num_exmpl = len(graph_dataset)
    num_train = int(num_exmpl * 0.8)
    # num_test = int(num_exmpl * 0.8)

    train_sampler = SubsetRandomSampler(th.arange(num_train))
    # Create a list to store the indices
    indices_list_train = []
    # Append the indices generated by the SubsetRandomSampler to the list
    for index_train in train_sampler:
        indices_list_train.append(index_train)
    
    print(indices_list_train)   # Print the list of train indices
    test_sampler = SubsetRandomSampler(th.arange(num_train, num_exmpl))
    indices_list_test = []
    for index_test in test_sampler:
        indices_list_test.append(index_test)
    print(indices_list_test)    # Print the list of test indices
    # val_sampler = SubsetRandomSampler(th.arange(num_test, num_exmpl))
    # indices_list_val = []
    # for index_val in val_sampler:
    #     indices_list_val.append(index_val)
    # print(indices_list_val)

    train_dataloader = GraphDataLoader(graph_dataset, sampler=train_sampler, batch_size=4, drop_last=False)
    test_dataloader = GraphDataLoader(graph_dataset, sampler=test_sampler, batch_size=4, drop_last=False)
    # val_dataloader = GraphDataLoader(graph_dataset, sampler=val_sampler, batch_size = 4, drop_last=False)
  
    # Create and train GCN Model
    # Create the model with given dimensions
    model1 = GCN(256, 128, 2)
    optimizer1 = th.optim.Adam(model1.parameters(), lr=0.01)
    trainer1 = ProcessModel(model1, train_dataloader, test_dataloader, optimizer1, num_epochs=50)
    trainer1.run()

