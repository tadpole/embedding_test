function [] = AROPE_CMD(input_file,output_file1,output_file2,d,order,weight)
edge_list = load(input_file);
edge_list = edge_list + 1;
A = sparse(edge_list(:,1),edge_list(:,2),1,max(max(edge_list)),max(max(edge_list)));
A = A + A';
order = order + 1;
weight_cell = cell(1,1);
weight = [1, weight];
weight_cell{1,1} = power(10, weight);
[U_cell,V_cell] = AROPE(A,d,order,weight_cell);
U = U_cell{1};
V = V_cell{1};
save(output_file1,'U');
save(output_file2,'V');


end
