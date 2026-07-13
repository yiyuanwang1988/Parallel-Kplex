//============================================================================
// Name        : splex-big.cpp
// Author      : JOEY
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <stdio.h>
#include <chrono>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <libgen.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <sstream>
#include <list>
#include <utility>
#include <stack>
#include <boost/thread.hpp>
#include <shared_mutex>
#include <mutex>
#include <atomic>
using namespace std;
#define LARGE_INT 999999999
/*Parameters*/

typedef struct ST_RandAccessList {
	int *vlist;
	int *vpos;
	int vnum;
	int capacity;
}RandAccessList;

int cnt_layers;
int avg_mindeg;
int *layers;
long long *sum_layers;

struct st
{
	int n,x;
	bool operator <(const st &a)const
	{
		if(a.x == x) return a.n < n; 
		return x < a.x;
	}
};

struct EDGE
{
	int v1;
	int v2;
};
EDGE* edge;

typedef struct {
    int sol_size;
    int *sol;
} SolutionArray;

char param_graph_file_name[1024]="/home/zhou/benchmarks/splex/2nd_dimacs/brock400_4.clq";
int param_s = 2;
int param_best = 999999;
int param_max_seconds = 120;
int param_cycle_iter = 1000;
unsigned int param_seed;
int thread_size = 10;

double alpha = 1.0;
int lambda = 100;
double gama = 0.8;

int org_vnum;
int org_enum;
int org_fmt;
int* org_v_edge_cnt;
int** org_v_adj_vertex;
int *org_vid;


int *red_orgid;
int *new_id;
//int *orgid_red;
//int* red_v_edge_cnt;
int** red_v_adj_vertex;


int** k_decompos;
int *k_cnt_decompos;
int *org_decompos;
int *new_edge_count;
int **new_adj_tbl;

int global_best_size = 0;
double global_best_time = 0.0;
double gloabl_avg;
int min_vnum;

int low_group_size = 10;
int up_group_size;
int group_num;
int *local_in_group;
int *local_in_remain;
int *local_in_cand;
int *v_in_tem;
int *cur_g_deg;
int *v_in_layer;
int *global_layer_weight;
int *deg_in_group;
int *avg_deg_group;
int *update_id_group;

vector<int> v_remain;
vector<vector<int>> v_group;

int * group_weight;
int* global_best_plex;


std::shared_mutex max_mutex;
std::shared_mutex group_mutex;
std::shared_mutex weight_mutex;
std::mutex printf_mutex;
std::atomic<bool> updated(false);

void printf_reslut(int num, double time, int flag){
	std::lock_guard<std::mutex> lock(printf_mutex);
	cout << param_s << " " << param_graph_file_name << " final " << num << " " << time << " "  << flag <<endl;
	exit(0);
}

void update_layer_weight(int * layer_weigt) {
	std::unique_lock<std::shared_mutex> lock(weight_mutex);
	for(int i = 0; i < cnt_layers; i++) {
		global_layer_weight[i] += layer_weigt[i];
		//printf("i %d weight %d mindeg %d\n", i, global_layer_weight[i], layers[i]);
	}
}

int select_start_vertex(int *freqs, int vnum) {
	int layer, best_layer, v, startv;
	vector<int> vec_cand;
	std::shared_lock<std::shared_mutex> lock(weight_mutex);
	best_layer = rand() % cnt_layers;
	for(int i = 0; i < cnt_layers % 100; i++) {
		layer = rand() % cnt_layers;
		if (global_layer_weight[layer] > global_layer_weight[best_layer]) best_layer = layer;
	}

	for (int i = 0; i < k_cnt_decompos[layers[best_layer]]; i++) {
		v = k_decompos[layers[best_layer]][i];
		if (v >= vnum) continue;
		vec_cand.push_back(v);
	}
	//printf("select_start_vertex min_vnum %d vnum %d\n", min_vnum, vnum);
	if (vec_cand.empty()) startv = rand() % vnum;
	else startv = vec_cand[rand() % vec_cand.size()];

	return startv;
}


void update_global(int value, double btime, int * best_solution) {
	std::unique_lock<std::shared_mutex> lock(max_mutex);
	if (global_best_size < value) {
		global_best_size = value;
		global_best_time = btime;
		for (int i = 0; i < global_best_size; i++) global_best_plex[i] = best_solution[i];
		//memcpy(global_best_plex, best_solution, sizeof(int) * org_vnum);
	}
	else if (global_best_size == value) {
		if (global_best_time > btime) {
			global_best_time = btime;
			for (int i = 0; i < global_best_size; i++) global_best_plex[i] = best_solution[i];
			//memcpy(global_best_plex, best_solution, sizeof(int) * org_vnum);
		}
	}
}

int is_update_global(int value) {
	std::shared_lock<std::shared_mutex> lock(max_mutex);
	if (global_best_size < value) return 1;
	else return 0;
}

int is_reduced_group(int deg) {
	std::shared_lock<std::shared_mutex> lock(max_mutex);
	if (global_best_size - param_s >= deg) return global_best_size - param_s;
	else return 0;
}


void update_group_weight(int id) {
	std::unique_lock<std::shared_mutex> lock(group_mutex);
    group_weight[id] += 2;
	/*printf("update group number %ld sel_group %d\n", v_group.size(), id);
	for (int i = 0; i < v_group.size(); i++) {
		printf("i %d size %ld weight %d\n", i, v_group[i].size(), group_weight[i]);
	}*/
}

pair<int, int> get_start_vertex(){
	int sel_group, group_size, randv, sum_weight, randw;
	std::shared_lock<std::shared_mutex> lock(group_mutex);
	sum_weight = 0;
	for (int i = 0; i < v_group.size(); i++) {
		sum_weight += group_weight[i];
	}
    randw = rand() % sum_weight;
	//printf("sum %d randw %d ", sum_weight, randw);
	for (int i = 0; i < v_group.size(); i++) {
		if (randw < group_weight[i]) {
			sel_group = i;
			break;
		}
		else {
			randw = randw - group_weight[i];
		}
	}
	for (int i = 0; i < v_group.size(); i++) {
		if (i == sel_group) continue;
		group_weight[i]++;
	}
	//sel_group = rand() % v_group.size();
	group_size = v_group[sel_group].size();
	randv = v_group[sel_group][rand() % group_size];
	return make_pair(sel_group, randv);
}

int is_v_in_group(int vertex) {
	std::shared_lock<std::shared_mutex> lock(group_mutex);
	if (local_in_group[vertex] == -1) return 0;
	else return 1;
}

int is_group_empty(){
	std::shared_lock<std::shared_mutex> lock(group_mutex);
	if (v_group.empty()) return 1;
	else return 0;
}

int is_update_group(int vnum) {
	std::shared_lock<std::shared_mutex> lock(group_mutex);
	if (min_vnum > vnum) return 1;
	else return 0;
} 

void remove_from_group(int i, int j, int vertex) {
	v_group[i].erase(v_group[i].begin() + j);
	local_in_group[vertex] = -1;
	deg_in_group[vertex] = 0;
}

void add_to_group(int i, int vertex) {
	if (i >= v_group.size()) {
        v_group.push_back(vector<int>()); // 添加一个新的空向量
		group_weight[i] = 1;
    }
	v_group[i].push_back(vertex);
	local_in_group[vertex] = i;
}

void update_degree_in_group (int i, int vnum) {
	int v1, v2;
	avg_deg_group[i] = 0;
	for (int j = 0; j < v_group[i].size(); j++) {
		v1 = v_group[i][j];
		deg_in_group[v1] = 0;
		for (int l = 0; l < new_edge_count[v1]; l++) {
			v2 = new_adj_tbl[v1][l];
			if (v2 >= vnum) break;
			if (local_in_group[v2] == i) deg_in_group[v1]++;
		}
		avg_deg_group[i] += deg_in_group[v1];
	}
	avg_deg_group[i] = avg_deg_group[i] / v_group[i].size();
}

void update_group(int vnum, int id, vector<int>& vec_temp){
	int v1, v2, id_group;
	vector<int> vec_cand;
	updated.store(true);
	{
		std::unique_lock<std::shared_mutex> lock(group_mutex);
		//printf("befor red size %ld\n", v_group.size());
		//for (int i = 0; i < v_group.size(); i++) printf("i %d size %ld weight %d\n", i, v_group[i].size(), group_weight[i]);
		if (min_vnum > vnum) min_vnum = vnum;
		for (int i = 0; i < v_group.size();) {
			for (int j = 0; j< v_group[i].size();) {
				v1 = v_group[i][j];
				if (v1 >= min_vnum) {
					remove_from_group(i, j, v1);
				}
				else j++;
			}
			if (v_group[i].size() < low_group_size) {
				//printf("erase i %d size %ld low %d\n", i, v_group[i].size(), low_group_size);
				while (!v_group[i].empty()) {
					int j = v_group[i].size() - 1;
					remove_from_group(i, j, v_group[i][j]);
				}
			}
			if(v_group[i].empty()) {
				if (v_group[i+1].empty()) group_weight[i] = 1;
				else group_weight[i] = group_weight[i+1];
				v_group.erase(v_group.begin() + i);
			}
			else {
				for (int j = 0; j < v_group[i].size(); j++) {
                	local_in_group[v_group[i][j]] = i;
            	}
				update_degree_in_group(i, min_vnum);
            	i++; 
			}
		}


		if (v_group.empty()) {
			for (int i = 0; i < vec_temp.size(); i++) {
				if (vec_temp[i] < min_vnum) {
					vec_cand.push_back(vec_temp[i]);
				}
			}
			if (vec_cand.size() >= low_group_size) {
				id_group = v_group.size();
				for (int j = 0; j < vec_cand.size(); j++) {
					add_to_group(id_group, vec_cand[j]);
				}
				update_degree_in_group(id_group, min_vnum);
			}
		}
		//printf("in id %d updated %d\n", id, updated.load());
		//printf("after red size %ld\n", v_group.size());
		//for (int i = 0; i < v_group.size(); i++) printf("i %d size %ld weight %d vnum %d\n", i, v_group[i].size(), group_weight[i], min_vnum);
	    
	}
	updated.store(false);
	//printf("out id %d updated %d\n", id, updated.load());
}

void update_group_partition(int vnum, vector<int>& vec_temp, int *freqs, long long *steps){
	int v, adjv, lastv, size, max_indeg, id_group, sum_freq, sum_v, sum_deg, avg_deg, group_size;
	long long sum_step, avg_step;
	vector<int> vec_cand;
	int flag = 0;
    
	std::unique_lock<std::shared_mutex> lock(group_mutex);
	sum_step = 0, sum_freq = 0;
	for (int i = 0; i < min_vnum; i++) {
		sum_step += steps[i];
		sum_freq += freqs[i];
	}
	avg_step = sum_step / (sum_freq + 1);
	for (int i = 0; i < v_group.size();) {
		for (int j = 0; j< v_group[i].size();) {
			v = v_group[i][j];
			if ((freqs[v] > 0) && (steps[v] > 0)) {
				if(((avg_step > steps[v]/freqs[v]) && (avg_deg_group[i] > deg_in_group[v])) || (v >= min_vnum)) {
					remove_from_group(i, j, v);
				}
				else j++;
			}
			else j++;
		}
		if (v_group[i].size() < low_group_size) {
			while (!v_group[i].empty()) {
				int j = v_group[i].size() - 1;
				remove_from_group(i, j, v_group[i][j]);
			}

		}
		if(v_group[i].empty()) {
			if (v_group[i+1].empty()) group_weight[i] = 1;
			else group_weight[i] = group_weight[i+1];
			v_group.erase(v_group.begin() + i);
		}
		else {
			for (int j = 0; j < v_group[i].size(); j++) {
                local_in_group[v_group[i][j]] = i;
            }
			update_degree_in_group(i, min_vnum);
            i++; 
		}
	}
	
	if (v_group.empty()) {
		for (int i = 0; i < vec_temp.size(); i++) {
			if (vec_temp[i] < min_vnum) {
				vec_cand.push_back(vec_temp[i]);
			}
		}
		if (vec_cand.size() >= low_group_size) {
			id_group = v_group.size();
			for (int j = 0; j < vec_cand.size(); j++) {
				add_to_group(id_group, vec_cand[j]);
			}
			update_degree_in_group(id_group, min_vnum);
		}
		flag = 1;
	}
	
	for (int i = 0; i < v_group.size(); i++) {
		if (v_group[i].size() > up_group_size) {
			if (v_group[i].size() % up_group_size == 0) group_size = v_group[i].size() / (v_group[i].size() / up_group_size);
			else group_size = v_group[i].size() / ((v_group[i].size() / up_group_size) + 1);
			
		}
		while ((v_group[i].size() > up_group_size)  && ((v_group[i].size() - up_group_size) > low_group_size)) {
			vec_cand.clear();
			for (int j = 0; j < min_vnum; j++) v_in_tem[j] = -1;
			for (int j = 0; j < v_group[i].size(); j++) v_in_tem[v_group[i][j]] = j;
	        
			while(vec_cand.size() < group_size) {
				for (int j = 0; j < v_group[i].size(); j++) {
					v = v_group[i][j];
					if (deg_in_group[v] < avg_deg_group[i]) break;
				}
				if (v_in_tem[v] != -1) {
					lastv = v_group[i].back();
					v_in_tem[lastv] = v_in_tem[v];
		            swap(v_group[i][v_in_tem[v]], v_group[i].back());
		            v_group[i].pop_back();
					vec_cand.push_back(v);
					v_in_tem[v] = -1;
				}
				
			    for (int j = 0; j < new_edge_count[v]; j++) {
					adjv = new_adj_tbl[v][j];
					if (adjv >= min_vnum) break;
			    	if ((local_in_group[adjv] == i) && (v_in_tem[adjv] != -1)) {
						lastv = v_group[i].back();
						v_in_tem[lastv] = v_in_tem[adjv];
		            	swap(v_group[i][v_in_tem[adjv]], v_group[i].back());
		            	v_group[i].pop_back();
						vec_cand.push_back(adjv);
						v_in_tem[adjv] = -1;
					}
					if (vec_cand.size() >= group_size) {	
						break;
					}
				}
				
			}
			
			if (vec_cand.size() >= group_size) {
				id_group = v_group.size();
				for (int j = 0; j < vec_cand.size(); j++) {
					add_to_group(id_group, vec_cand[j]);
				}
				update_degree_in_group(id_group, min_vnum);
			}
			else {
				for (int j = 0; j < vec_cand.size(); j++) {
					local_in_group[vec_cand[j]] = i;
			        v_group[i].push_back(vec_cand[j]);
				}
			}
			
		}
		update_degree_in_group(i, min_vnum);
	}
	
	if (!flag) {
		vec_cand.clear();
		memset(update_id_group, 0, v_group.size() * sizeof(int));
		for (int i = 0; i < vec_temp.size(); i++) {
			v = vec_temp[i];
			if (local_in_group[v] != -1 || v >= min_vnum) continue;
			memset(cur_g_deg, 0, min_vnum * sizeof(int));
			size = 0;
			for (int j = 0; j < new_edge_count[v]; j++) {
				adjv = new_adj_tbl[v][j];
				if (adjv >= min_vnum) break;
				if (local_in_group[adjv] != -1) {
					cur_g_deg[local_in_group[adjv]]++;
					if (size < local_in_group[adjv]) size = local_in_group[adjv];
				}
 			}
			max_indeg = 0;
			id_group = 0;
			int flag1 = 0;
			
			for (int j = 0; j < size; j++) {
				if ((avg_step <= steps[v]/(freqs[v] + 1)) && (avg_deg_group[j] <= cur_g_deg[j])) {
					id_group = j;
					flag1 = 1;
					break;
				}
			}
			if (!flag1) {
				vec_cand.push_back(v);
				continue;
			}
			add_to_group(id_group, v);
			if (update_id_group[id_group] == 0) update_id_group[id_group] = 1;
		}
        for (int i = 0; i < v_group.size(); i++) {
			if (update_id_group[i] == 1) update_degree_in_group(i, min_vnum);
		}
		if (vec_cand.size() >= low_group_size) {
			id_group = v_group.size();
			for (int j = 0; j < vec_cand.size(); j++) {
				add_to_group(id_group, vec_cand[j]);
			}
			update_degree_in_group(id_group, min_vnum);
		}
	}
	
}

void printf_group(int x, int y, int id, int vnum) {
	std::shared_lock<std::shared_mutex> lock(group_mutex);
	printf("group number %ld met %d sel_group %d id %d vnum %d\n", v_group.size(), x, y, id, vnum);
	/*for (int i = 0; i < v_group.size(); i++) {
		printf("i %d size %ld weight %d \n", i, v_group[i].size(), group_weight[i]);
	}*/
}


int load_clq_instance(char* filename) {
		ifstream infile(filename);
		char line[1024];
		char tmps1[1024];
		char tmps2[1024];
		if (!infile.is_open()) {
			fprintf(stderr, "Can not find file %s\n", filename);
			return 0;
		}
	
		infile.getline(line, 1024);
		while (line[0] != 'p')	infile.getline(line, 1024);
		sscanf(line, "%s %s %d %d", tmps1, tmps2, &org_vnum, &org_enum);
		edge = (EDGE*)malloc(org_enum * sizeof(EDGE));
	
		int ecnt = 0;
		org_v_edge_cnt = new int[org_vnum];
		memset(org_v_edge_cnt, 0, org_vnum * sizeof(int));
		while (infile.getline(line, 1024)) {
			int v1, v2;
			if (strlen(line) == 0)
				continue;
			if (line[0] != 'e')
				fprintf(stderr, "ERROR in line %d\n", ecnt + 1);
			sscanf(line, "%s %d %d", tmps1, &v1, &v2);
			v1--, v2--;
			org_v_edge_cnt[v1]++;
			org_v_edge_cnt[v2]++;
			edge[ecnt].v1 = v1;
			edge[ecnt].v2 = v2;
			ecnt++;
		}
		assert(org_enum == ecnt);
		org_fmt = 0;
		org_vid = new int[org_vnum];
		org_v_adj_vertex = new int*[org_vnum];
		for (int i = 0; i < org_vnum; i++) {
			org_v_adj_vertex[i] = new int[org_v_edge_cnt[i]];
			org_v_edge_cnt[i] = 0;
		}
		int u, v;
		for (int i = 0; i < org_enum; ++i)
		{
			u = edge[i].v1;
			v = edge[i].v2;
			org_v_adj_vertex[u][org_v_edge_cnt[u]++] = v;
			org_v_adj_vertex[v][org_v_edge_cnt[v]++] = u;
		}
		for (int i = 0; i < org_vnum; i++) {
			org_vid[i] = i + 1;
		}
		free(edge);
		return 1;
}
	

int load_snap_instance(char *filename){
		ifstream infile(filename);
		char line[1024];
		vector<pair<int,int> > *pvec_edges = new vector<pair<int, int> >();
		const int CONST_MAX_VE_NUM = 9999999;
		if (infile.is_open() == false) {
			fprintf(stderr, "Can not find file %s\n", filename);
			return 0;
		}
		int max_id = 0;
		int from, to;
		//int max_id = 0;
		while (infile.getline(line, 1024)){
			char *p = line;
			while (*p ==' ' && *p !='\0') p++;
			if (*p != '#'){
				stringstream ss(line);
				ss >> from >> to;
				//printf("%d %d \n", from, to);
				pvec_edges->push_back(make_pair(from, to));
				if (from > max_id)
					max_id = from;
				else if (to > max_id)
					max_id = to;
			}
		}
		int *newid = new int[max_id+1];
		org_vid = new int[CONST_MAX_VE_NUM];
			//init the newid map
		for (int i = 0; i < max_id+1; i++){
			newid[i] = -1;
		}
	
		int v_num = 0;
		//count edges,resign ids from 1...v_num
		for (int i = 0; i < (int)pvec_edges->size(); i++){
			from = (pvec_edges->at(i)).first;
			if (newid[from] == -1){
				newid[from] = v_num;
				org_vid[v_num] = from;
				v_num++;
			}
			(pvec_edges->at(i)).first = newid[from];
			to = (pvec_edges->at(i)).second;
			if (newid[to] == -1) {
				newid[to] = v_num;
				org_vid[v_num] = to;
				v_num++;
			}
			(pvec_edges->at(i)).second = newid[to];
		}
	//	sort(pvec_edges->begin(), pvec_edges->end());
	//	for (int i = 0; i < pvec_edges->size(); i++){
	//		printf("%d %d\n", pvec_edges->at(i).first, pvec_edges->at(i).second);
	//	}
		org_vnum = v_num;
		int *estimate_edge_cnt = new int[org_vnum];
		memset(estimate_edge_cnt, 0, sizeof(int) * org_vnum);
		for (int i = 0; i < (int)pvec_edges->size(); i++){
			from = (pvec_edges->at(i)).first;
			to = (pvec_edges->at(i)).second;
			estimate_edge_cnt[from]++;
			estimate_edge_cnt[to]++;
		}
		org_enum = 0;
		org_v_edge_cnt = new int[org_vnum];
		memset(org_v_edge_cnt, 0, sizeof(int) * org_vnum);
		org_v_adj_vertex = new int*[org_vnum];
		for (int i = 0; i < (int)pvec_edges->size(); i++){
			from = (pvec_edges->at(i)).first;
			to = (pvec_edges->at(i)).second;
			if (from == to) continue; //self-edges are abandoned
			if (org_v_edge_cnt[from] == 0){
				org_v_adj_vertex[from] = new int[estimate_edge_cnt[from]];
			}
			int exist1 = 0;
			for (int j = 0; j < org_v_edge_cnt[from]; j++){
				if (org_v_adj_vertex[from][j] == to){
					exist1 = 1;
					break;
				}
			}
			if (!exist1){
				org_v_adj_vertex[from][org_v_edge_cnt[from]] = to;
				org_v_edge_cnt[from]++;
				org_enum++;
			}
	
			if (org_v_edge_cnt[to] == 0){
				org_v_adj_vertex[to] = new int[estimate_edge_cnt[to]];
			}
			int exist2 = 0;
			for (int j = 0; j < org_v_edge_cnt[to]; j++){
				if (org_v_adj_vertex[to][j] == from){
					exist2 = 1;
					break;
				}
			}
			if (!exist2){
				org_v_adj_vertex[to][org_v_edge_cnt[to]] = from;
				org_v_edge_cnt[to]++;
				org_enum++;
			}
		}
		assert(org_enum %2 == 0);
		org_enum = org_enum/2;
		printf("load SNAP graph %s with vertices %d, edges %d (directed edges %d)\n",
				basename(param_graph_file_name), org_vnum, org_enum, (int)pvec_edges->size());
		//print_org_graph();
		//build v_
		delete[] newid;
		delete[] estimate_edge_cnt;
		delete pvec_edges;
		return 1;
}

int load_metis_instance(char* filename){
	ifstream infile(filename);
	string line;
	
	if (infile.is_open() == false) {
		fprintf(stderr, "Can not find file %s\n", filename);
		return 0;
	}
	org_fmt = 0;
	//ignore comment
	getline(infile, line);
	while(line.length()==0 || line[0] == '%')
		getline(infile, line);
	stringstream l1_ss(line);
	l1_ss >> org_vnum >> org_enum >> org_fmt;
	cout << line << endl;
	if (org_fmt == 100){
		cerr << "self-loops and/or multiple edges need to be considered";
		return 0;
	}
	/*allocate graph memory*/
	org_v_edge_cnt = new int[org_vnum];
	org_v_adj_vertex = new int*[org_vnum];
	org_vid = new int[org_vnum];
	
	int v_no = 0;
	int *vlst = new int[org_vnum];
	int n_adj = 0;
	//ignore the char
	//read the end of first line
	//infile.getline(line, LINE_LEN);
	while (getline(infile, line) && v_no < org_vnum){
		if (infile.fail()){
			fprintf(stderr, "Error in read file, vno %d\n",v_no);
			exit(-1);
		}
		stringstream ss(line);
		int ve_adj;
	
		n_adj = 0;
	 	while (ss >> ve_adj){
			vlst[n_adj++] = ve_adj-1;
		}
		org_v_edge_cnt[v_no] = n_adj;
		if (n_adj == 0)
			org_v_adj_vertex[v_no] = NULL;
		else{
			org_v_adj_vertex[v_no] = new int[n_adj];
			memcpy(org_v_adj_vertex[v_no], vlst, sizeof(int) * n_adj);
		}
		org_vid[v_no] = v_no+1;
		v_no++;
	}
	assert(v_no == org_vnum);
	printf("load metis graph %s with vertices %d, edges %d \n",
				basename(param_graph_file_name), org_vnum, org_enum);
		//debug
	delete[] vlst;
	infile.close();
	return 1;
}

void showUsage(){
	fprintf(stderr, "ParaKplex -f <filename> -k <parameter> -t <max seconds> [-o optimum object] [-n threads]\n");
}

void read_params(int argc, char **argv){
		int hasFile = 0;
		int hasTimeLimit = 0;
		int hasS = 0;
		for (int i = 1; i < argc; i+=2){
			if (argv[i][0] != '-' || argv[i][2] != 0){
				showUsage();
				exit(0);
			}else if (argv[i][1] == 'f'){
				strncpy(param_graph_file_name, argv[i+1],1000);
				hasFile = 1;
			}else if(argv[i][1] == 'k'){
				param_s = atoi(argv[i+1]);
				if (param_s >= 1)	hasS = 1;
			}else if (argv[i][1] == 'o'){
				param_best = atoi(argv[i+1]);
			}else if (argv[i][1] == 'c'){
				param_seed = atoi(argv[i+1]);
			}else if(argv[i][1] == 't'){
				param_max_seconds = atoi(argv[i+1]);
				hasTimeLimit = 1;
			}else if(argv[i][1] == 'n'){
				thread_size = atoi(argv[i+1]);
				if (thread_size < 1){
					fprintf(stderr, "Invalid thread count: %s\n", argv[i+1]);
					exit(1);
				}
			}
		}
		//check parameters
		if (!hasFile){
			fprintf(stderr,"No file name\n");
			showUsage();
			exit(1);
		}
		if (!hasTimeLimit){
			fprintf(stderr,"No time limit \n");
			showUsage();
			exit(1);
		}
		if (!hasS){
			fprintf(stderr,"No paramete s\n");
			showUsage();
			exit(1);
		}
}
	
const char* file_suffix(char* filename){
	const char *dot = strrchr(filename, '.');
	if(!dot || dot == filename) return "";
	return dot + 1;
}

void load(int argc, char **argv)
{
	read_params(argc, argv);
	int load = 0;
	const char* fileext = file_suffix(param_graph_file_name);
	//printf("%s\n",fileext);
	if (0 == strcmp(fileext, "graph")){
		load = load_metis_instance(param_graph_file_name);
	}else if(0 == strcmp(fileext, "txt")){
		load = load_snap_instance(param_graph_file_name);
	}else if(0 == strcmp(fileext, "mtx")){
		load = load_clq_instance(param_graph_file_name);
	}
	else {
		load = load_clq_instance(param_graph_file_name);
	}
	if (load != 1){
		printf("%d %s failed in loading graph %s\n",param_s,param_graph_file_name,param_graph_file_name);
		exit(-1);
	}
} 

class Splex
{
public:
	//char param_graph_file_name[1024]="/home/zhou/benchmarks/splex/2nd_dimacs/brock400_4.clq";
	//int param_s = 2;
	//int param_best = 9999;
	//int param_max_seconds = 120;
	//int param_cycle_iter = 1000;
	//unsigned int param_seed;
	int reduction_num = 0;
	int *mark;
	int *threshold;
	//int *rmflag; 
	int *deposit;
	//int *remain_edge_count;
	//int *org_id;
	//int *last_freq;
	//int *last_momentum;


	/*original graph, the structure is kept the same,
	 * but the id of vertices are renumbered, the original
	 * id can be retrievaled by org_vid*/
	/*int org_vnum;
	int org_enum;
	int org_fmt;
	int* org_v_edge_cnt;
	int** org_v_adj_vertex;
	int** cnt_adj_tbl; 
	int** new_adj_tbl;
	int *org_vid;	//The real id of each original vertex*/
	
	/*reduced graph*/
	int red_vnum;
	int red_enum;
	//int *red_orgid; /*the corresponding id in original graph*/
	//int** past_v_adj_vertex;
	int* red_v_edge_cnt;
	//int** red_v_adj_vertex;
	int red_min_deg ;
	int *freq;
	int *momentum;
	int *tabu_add;
	int cur_iter;
	int *cur_c_deg;
	//int *cur_c_consat; /*cur_c_consat[v] Then number of saturated neighboors of v*/
	//int cur_c_satu_num; //the number of saturate vertices
	int *is_in_c;

	
	RandAccessList *cur_splex;
	RandAccessList *cur_cand;
	RandAccessList *cur_remain;
	//RandAccessList *cur_satured;
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point stime;
	std::chrono::steady_clock::time_point etime;
	double totime = 0;

	
	/*final result*/
	int best_size;
	int best_size2;
	int* best_plex;
	std::chrono::steady_clock::time_point best_time;
	int best_found_iter;
	std::chrono::steady_clock::time_point total_time;
	unsigned int rand_seed = 0,rand_seed_record = 0; 

	

	struct timespec start_time1;
	struct timespec best_time1;
	struct timespec total_time1;
	struct timespec stime1;
	struct timespec etime1;

	double the_best_time;

	long long freq_inc;
	long long freq_old;
	long long *step_in_solution;
	long long *add_step;
	long long iter_step;

    int init_met;
	int	start_group;

	int time_update = 0;
	int restart_pass = 0;
	int *local_layer_weight;
	vector<int> ver_inc_deg0;

	int* profit;
	

	/*struct EDGE
	{
		int v1;
		int v2;
	};
	EDGE* edge;*/
	
	void set_seed(unsigned int new_seed)
	{
		rand_seed = new_seed;
		rand_seed_record = rand_seed;
	}
	
	RandAccessList* ral_init(int capacity) {
		RandAccessList *ral = new RandAccessList;
		ral->vlist = new int[capacity];
		ral->vpos = new int[capacity];
		ral->vnum = 0;
		ral->capacity = capacity;
		for (int i = 0; i < capacity; i++) {
			ral->vpos[i] = capacity;
		}
		return ral;
	}
	
	void ral_add(RandAccessList *ral, int vid) {
		//assert(ral->vpos[vid] < ral->vnum);
		ral->vlist[ral->vnum] = vid;
		ral->vpos[vid] = ral->vnum;
		ral->vnum++;
	}
	void ral_delete(RandAccessList *ral, int vid) {
		//assert(ral->vpos[vid] < ral->vnum);
		int last_id = ral->vlist[ral->vnum - 1];
		int id_pos = ral->vpos[vid];
		ral->vlist[id_pos] = last_id;
		ral->vpos[last_id] = id_pos;
		ral->vnum--;
		//	ral->vpos[vid] = ral->vnum; /*It is not obligatory*/
	}
	
	void ral_clear(RandAccessList *ral) {
		ral->vnum = 0;
	}

	void ral_reset(RandAccessList *ral, int capacity) {
		for (int i = 0; i < capacity; i++) {
			ral->vlist[i] = i;
		    ral->vpos[i] = i;
		}
		ral->vnum = capacity;
	}
	
	
	void ral_release(RandAccessList *ral) {
		delete[] ral->vlist;
		delete[] ral->vpos;
		delete ral;
	}
	
	static int cmpfunc(const void * a, const void * b)
	{
		return (*(int*)a - *(int*)b);
	}
	
	void ral_showList(RandAccessList *ral, FILE *f) {
		fprintf(f, "Total %d: ", ral->vnum);
		int *tmp_lst = new int[ral->capacity];
		memcpy(tmp_lst, ral->vlist, ral->vnum * sizeof(int));
		qsort(tmp_lst, ral->vnum, sizeof(int), cmpfunc);
		for (int i = 0; i < ral->vnum; i++) {
			fprintf(f, "%d ", tmp_lst[i]);
		}
		fprintf(f, "\n");
	}
	
	/*debug*/
	void print_reduced_graph(){
		cout << red_vnum << " " << red_enum << endl;
		if (red_vnum <= 50){
			for (int v = 0; v < red_vnum; v++){
				printf("v %d (%d):", v, red_v_edge_cnt[v]);
				for (int i = 0; i < red_v_edge_cnt[v]; i++){
					cout << red_v_adj_vertex[v][i] << " ";
				}
				cout << endl;
			}
		}
	}
	
	void print_org_graph(){
		cout << org_vnum << " " << org_enum <<" "<< org_fmt << endl;
		for (int v = 0; v < org_vnum; v++){
			printf("v %d (%d):", v, org_v_edge_cnt[v]);
			for (int i = 0; i < org_v_edge_cnt[v]; i++){
				cout << org_v_adj_vertex[v][i] << " ";
			}
			cout << endl;
		}
	}
	
	/**
	 * load instances from  2nd Dimacs competetion
	 *
	 */
	
	/*load instances from  Stanford Large Network Dataset Collection
	 * URL: http://snap.stanford.edu/data/*/
	
	
	/*load instances of metis format from
	 * instances are download from
	 * http://www.cc.gatech.edu/dimacs10/downloads.shtml
	 */
	
	void print_current_solution(){
		printf("current solution \n");
		printf("size:%d \n",cur_splex->vnum);
		int *tmp_lst= new int[cur_splex->vnum];
		memcpy(tmp_lst, cur_splex->vlist, cur_splex->vnum * sizeof(int));
		qsort(tmp_lst, cur_splex->vnum, sizeof(int), cmpfunc);
		for(int i = 0;i < cur_splex->vnum; i++){
			printf("%d(%d) ", tmp_lst[i], cur_c_deg[tmp_lst[i]]);
		}
		printf("\n");
		delete[] tmp_lst;
	}
	void print_current_contex(){
		print_current_solution();
		printf("current candidate size %d :\n", cur_cand->vnum);
		for (int i = 0; i< cur_cand->vnum; i++){
			int v = cur_cand->vlist[i];
			printf("%d(%d): ", v, cur_c_deg[v]);
			for (int j = 0; j < red_v_edge_cnt[v]; j++){
				int vadj = red_v_adj_vertex[v][j];
				if (is_in_c[vadj])
					printf("%d ", vadj);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	/*recursively remove all the vertices with degree less or equal than reduce_deg,
	 * reconstructing the reduced graph  */
	/*void reduce_graph(int reduce_deg){
		//printf("----------------------\n");
		int rm_cnt = 0;
		//int *rmflag = new int[red_vnum];
		//int *remain_edge_count = new int[red_vnum];
		//Avoid unnecessary reduction
		if (red_min_deg > reduce_deg){
			return;
		}
		for(int i = 0;i < red_vnum;i++) 
		{
			cnt_adj_tbl[i] = new_adj_tbl[i];
			past_v_adj_vertex[i] = red_v_adj_vertex[i];
		}
		memcpy(remain_edge_count, red_v_edge_cnt, sizeof(int) * red_vnum);
		memset(rmflag, 0, sizeof(int) * red_vnum);
		queue<int> rm_que;
		for (int idx = 0; idx < red_vnum; idx++){
			if (remain_edge_count[idx] <= reduce_deg){
				rmflag[idx] = 1;
				rm_que.push(idx);
				rm_cnt++;
			}
		}
		while(!rm_que.empty()){
			int idx = rm_que.front();
			rm_que.pop();
			for (int i = 0; i < red_v_edge_cnt[idx]; i++){
				int adjv = red_v_adj_vertex[idx][i];
				remain_edge_count[adjv]--;
				
				if (!rmflag[adjv] && remain_edge_count[adjv] <= reduce_deg){
					rmflag[adjv] = 1;
					rm_que.push(adjv);
					rm_cnt++;
				}
			}
		}
		 
		//rebuild the reduced graph
		int rest_vnum = red_vnum - rm_cnt;
		//int *new_id = new int[red_vnum];
		//Mapp the vertex id to the original id(in the initial graph) 
		//int *org_id = new int[rest_vnum];
		int count = 0; //count the rest vertices
		//int *last_freq = new int[rest_vnum];
		//resign new id to the rest of the vertices
		//int *last_momentum = new int[rest_vnum];
		for (int idx = 0; idx < red_vnum; idx++){
			if (!rmflag[idx]){
				new_id[idx] = count;
				org_id[count] = red_orgid[idx];
				last_freq[count] = freq[idx];
				last_momentum[count] = momentum[idx];
				count++;
			}
		}
		 
		int min_deg = LARGE_INT;
		int n_edges = 0;
		//int **new_adj_tbl = new int*[rest_vnum];
		//int *new_edge_count = new int[rest_vnum];
		for (int idx_prev = 0; idx_prev < red_vnum; idx_prev++){
			if (rmflag[idx_prev]){
				//delete[] red_v_adj_vertex[idx_prev];
				continue;
			}
			int idx_new = new_id[idx_prev];
			//swap(new_adj_tbl[idx_new],new_adj_tbl[idx_prev]);
			new_adj_tbl[idx_new] = cnt_adj_tbl[idx_prev];
			new_edge_count[idx_new] = remain_edge_count[idx_prev];
			int cnt = 0;
			for (int i = 0; i < red_v_edge_cnt[idx_prev]; i++){
				int vi_adj = red_v_adj_vertex[idx_prev][i];
				if (!rmflag[vi_adj]){
					new_adj_tbl[idx_new][cnt++] = new_id[vi_adj];
					n_edges++;
				}
			}
			if (new_edge_count[idx_new] < min_deg)
				min_deg = new_edge_count[idx_new];
			//printf("%d %d %d\n",idx_prev,cnt,remain_edge_count[idx_prev]);
			assert(cnt == remain_edge_count[idx_prev]);
			//swap(red_v_adj_vertex[idx_new],red_v_adj_vertex[idx_prev]);
			//delete[] red_v_adj_vertex[idx_prev];
		}
		assert(n_edges % 2 == 0);
		//assign to the new graph
		//delete[] red_v_adj_vertex;
		for(int i = 0;i < red_vnum;i++) 
		{
			if(!rmflag[i])
				red_v_adj_vertex[new_id[i]] = past_v_adj_vertex[i];
	    }
		swap(red_v_adj_vertex,new_adj_tbl);
		//red_v_adj_vertex = new_adj_tbl;
		//delete[] red_v_edge_cnt;
		swap(red_v_edge_cnt,new_edge_count);
		//red_v_edge_cnt = red_v_edge_cnt;
	
		red_vnum = rest_vnum;
		red_enum = n_edges/2;
	
		//reset the minimum deg
		red_min_deg = min_deg;
	
		//reset the original id map
		//delete[] red_orgid;
		swap(red_orgid,org_id);
		//red_orgid = org_id;
	
		//reset the last join record
		//delete[] freq;
		swap(freq,last_freq);
		//freq = last_freq;
	
		//delete[] momentum;
		swap(momentum,last_momentum);
		//momentum = last_momentum;
	
		//delete[] new_id;
		//delete[] rmflag;
		//delete[] remain_edge_count;
	
	}*/

	/*reinitial the data for a new start of the algorithm*/
	void init_search(int thread_id){
		/*copy the graph to reduce graph*/
		//red_vnum = org_vnum;
		red_enum = org_enum;
		//red_orgid = new int[red_vnum];
		
		//rmflag = new int[red_vnum];
		//remain_edge_count = new int[red_vnum];
		
		//new_id = new int[red_vnum];
		//org_id = new int[red_vnum];
		//last_freq = new int[red_vnum];
		//last_momentum = new int[red_vnum];
		//new_edge_count = new int[red_vnum];
		
		
		//red_v_edge_cnt = new int[red_vnum];
		//red_v_adj_vertex = new int*[red_vnum];
		
		
		//new_adj_tbl = new int*[red_vnum];
		//cnt_adj_tbl = new int*[red_vnum];
		//past_v_adj_vertex = new int*[red_vnum];
		//rmflag = new int [red_vnum]; 
		/*red_min_deg = LARGE_INT;
		memcpy(red_v_edge_cnt, org_v_edge_cnt, sizeof(int) * org_vnum);
		for (int v = 0; v < red_vnum; v++){
			red_orgid[v] = v;
			red_v_adj_vertex[v] = new int[red_v_edge_cnt[v]];
			new_adj_tbl[v] = new int[red_v_edge_cnt[v]];
			memcpy(red_v_adj_vertex[v], org_v_adj_vertex[v], sizeof(int) * org_v_edge_cnt[v]);
			if (red_v_edge_cnt[v] < red_min_deg)
				red_min_deg = red_v_edge_cnt[v];
		}*/
	   

		/*init search data*/
		cur_iter = 0;
		iter_step = 0;
		mark = new int[red_vnum];
		cur_c_deg = new int[red_vnum];
		is_in_c = new int[red_vnum];
		memset(cur_c_deg, 0, sizeof(int) * red_vnum);
		memset(is_in_c, 0, sizeof(int) * red_vnum);
		freq = new int[red_vnum];
		memset(freq, 0, sizeof(int) * red_vnum);
		momentum = new int[red_vnum];
		memset(momentum, 0, sizeof(int) * red_vnum);
		step_in_solution = new long long [red_vnum];
		memset(step_in_solution, 0, sizeof(long long) * red_vnum);
		add_step = new long long[red_vnum];
		memset(add_step, 0, sizeof(long long) * red_vnum);
		local_layer_weight = new int[cnt_layers];

		profit = new int[red_vnum];
		
	
		cur_splex = ral_init(red_vnum);
		cur_cand = ral_init(red_vnum);
		cur_remain = ral_init(red_vnum);
		ral_reset(cur_remain, red_vnum);
	
		tabu_add = new int[red_vnum];
	
		/*init best found data*/
		best_size = 0;
		/*We could reduce this size*/
		best_plex = new int[red_vnum];
		//best_time = 0;
		best_found_iter = 0;
	
		srand(thread_id);

	    //start_time = std::chrono::steady_clock::now();
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start_time1);
	    //srand((unsigned int)param_seed);
      
		threshold = new int[red_vnum];
		deposit = new int[red_vnum];

		red_min_deg = 1;
		init_met = 0;
        
	}
	
	void restart_search(){
		
		memset(cur_c_deg, 0, sizeof(int) * red_vnum);
		memset(is_in_c, 0, sizeof(int) * red_vnum);
		memset(add_step, 0, sizeof(long long) * red_vnum);
		ral_clear(cur_splex);
		ral_clear(cur_cand);
		ral_reset(cur_remain, red_vnum);
		memset(tabu_add, 0, sizeof(int) * red_vnum);
		for (int i = 0; i < red_vnum; ++i) {
			threshold[i] = 1;
			deposit[i] = 1;
		}
	}

	void whether_update_group(int thread_id) {
		long long sumfreq = 0;
		vector<int> vec_M1;
		int freq_thres;

		for (int i = 0; i < red_vnum; i++) {
			sumfreq += freq[i];
		}
		freq_inc = sumfreq - freq_old;

		if (restart_pass % 20 == 0) {
			if (((time_update * 1.0) / restart_pass) < 0.5) alpha = (0.125 > alpha / 2.0) ? 0.125 : (alpha / 2.0);
			else alpha = (2.0 < alpha * 2.0) ? 2.0 : (alpha * 2.0);
		}
		freq_thres = int (alpha * red_vnum);
		if (freq_inc > freq_thres) {
			for (int i = 0; i < best_size; i++) {
				int v = new_id[best_plex[i]];
				vec_M1.push_back(v);
				
			}
			if(!is_group_empty()) {
				update_group_partition(red_vnum, vec_M1, freq, step_in_solution);
				time_update++;
			}
			freq_old = sumfreq;
		}
		
	}

	void free_memory() {

		delete[] cur_c_deg;
		delete[] is_in_c;
		delete[] freq;
		delete[] momentum;
		delete[] threshold;
		delete[] deposit;
		delete[] red_v_edge_cnt;
		delete[] tabu_add;
		delete[] best_plex;
		delete[] step_in_solution;
		delete[] add_step;
		delete[] local_layer_weight;
		
	}

	
	int bms_thre_pro(int v) {
		int count = 0;
		for (int i = 0; i < 50; i++) {
			if (freq[v] > freq[rand_r(&rand_seed) % red_vnum]) {
				count++;
			}
		}
		return count;//若count很大说明freq很小，则threshold不++
	}
	
	void add_cur_vertex(int v){
		assert(!is_in_c[v]);
		deposit[v] = 0;
		is_in_c[v] = 1;
		freq[v]++;
		add_step[v] = iter_step;
		//printf("v %d add sol a\n", v);
		ral_add(cur_splex, v);
		++threshold[v];
		if (threshold[v] >= 3) {
			int countttt = bms_thre_pro(v);
			if (countttt > 40) {
				threshold[v] = 1;
			}
			else {
				threshold[v] = 0;
			}
		}
		if (cur_c_deg[v] > 0){
			//printf("v %d del can a\n", v);
			ral_delete(cur_cand, v);
		}
		if (cur_c_deg[v] == 0) {
			//printf("v %d del rem a\n", v);
			ral_delete(cur_remain, v);
		}
	
		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int adjv = red_v_adj_vertex[v][i];
			cur_c_deg[adjv]++;
			++deposit[adjv];
			++momentum[adjv];
			profit[adjv]++;
			if (!is_in_c[adjv] && cur_c_deg[adjv] == 1){
				//printf("v %d add can a\n", adjv);
				ral_add(cur_cand, adjv);
				//printf("v %d del rem a\n", adjv);
				ral_delete(cur_remain, adjv);
			}
		}

		// printf("add1 slo %d cand %d remain %d red_vnum %d\n", cur_splex->vnum, cur_cand->vnum, cur_remain->vnum, red_vnum);
		// for (int i = 0; i < cur_remain->vnum; i++) {
		// 	int v = cur_remain->vlist[i];
		// 	if (is_in_c[v]) printf("!!!error!!!\n");
		// }

	}

	void add_cur_vertex_profit(int v) {
		assert(!is_in_c[v]);

		is_in_c[v] = 1;
		ral_add(cur_splex, v);
		if (cur_c_deg[v] > 0) {
			ral_delete(cur_cand, v);
		}

		for (int i = 0; i < red_v_edge_cnt[v]; i++) {
			int adjv = red_v_adj_vertex[v][i];
			cur_c_deg[adjv]++;
			profit[adjv]++;
			if (!is_in_c[adjv] && cur_c_deg[adjv] == 1) {
				ral_add(cur_cand, adjv);
			}
		}
	}
	
	void add_without_change_deposit(int v) {
		assert(!is_in_c[v]);
	
		is_in_c[v] = 1;
		freq[v]++;
		add_step[v] = iter_step;
		//printf("v %d add sol aw\n", v);
		ral_add(cur_splex, v);
		if (cur_c_deg[v] > 0) {
			//printf("v %d del can aw\n", v);
			ral_delete(cur_cand, v);
		}
		if (cur_c_deg[v] == 0) {
			//printf("v %d del rem aw\n", v);
			ral_delete(cur_remain, v);
		}
	
		for (int i = 0; i < red_v_edge_cnt[v]; i++) {
			int adjv = red_v_adj_vertex[v][i];
			cur_c_deg[adjv]++;
			++momentum[adjv];
			if (!is_in_c[adjv] && cur_c_deg[adjv] == 1) {
				//printf("v %d add cand aw\n", adjv);
				ral_add(cur_cand, adjv);
				//printf("v %d del rem aw\n", adjv);
				ral_delete(cur_remain, adjv);
			}
		}

		// printf("add2 slo %d cand %d remain %d\n", cur_splex->vnum, cur_cand->vnum, cur_remain->vnum);
		// for (int i = 0; i < cur_remain->vnum; i++) {
		// 	int v = cur_remain->vlist[i];
		// 	if (is_in_c[v]) printf("!!!error!!!\n");
		// }
	}
	
	void remove_cur_vertex(int v){
		//printf("v %d is_in_c %d\n", v, is_in_c[v]);
		assert(is_in_c[v]);
		deposit[v] = 0;
		is_in_c[v] = 0;
		freq[v]++;
		step_in_solution[v] += iter_step - add_step[v];
		ral_delete(cur_splex, v);
		if (cur_c_deg[v] > 0){
			ral_add(cur_cand, v);
		}
		if (cur_c_deg[v] == 0) {
			ral_add(cur_remain, v);
		}
	
		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int adjv = red_v_adj_vertex[v][i];
			cur_c_deg[adjv]--;
			--momentum[adjv];
			if (!is_in_c[adjv] && cur_c_deg[adjv] == 0){
				ral_delete(cur_cand, adjv);
				ral_add(cur_remain, adjv);
			}
			if (is_in_c[adjv] && cur_c_deg[adjv] == 0) {
				ver_inc_deg0.push_back(adjv);
			}
		}

		// printf("remove slo %d cand %d remain %d\n", cur_splex->vnum, cur_cand->vnum, cur_remain->vnum);
		// for (int i = 0; i < cur_remain->vnum; i++) {
		// 	int v = cur_remain->vlist[i];
		// 	if (is_in_c[v]) printf("!!!error!!!\n");
		// }
	}
	
	#define Is_Saturated(v) (cur_c_deg[v] == cur_splex->vnum - param_s)
	#define Is_Overflow(v) (cur_c_deg[v] < cur_splex->vnum - param_s)
	
	int get_saturate_size(){
		int size = 0;
		for (int i = 0; i < cur_splex->vnum; i++){
			int vin = cur_splex->vlist[i];
			if (Is_Saturated(vin)){
				size++;
			}
		}
		return size;
	}
	
	void record_best(){
		best_size = cur_splex->vnum;
		if (best_size < 50) {
			best_size2 = 50;
		}
		else {
			best_size2 = best_size;
		}

		memset(local_layer_weight, 0, sizeof(int) * cnt_layers);
		for (int i = 0; i < cur_splex->vnum; i++){
			int v = cur_splex->vlist[i];
			best_plex[i] = red_orgid[v];
			local_layer_weight[v_in_layer[v]]++;
		}
		update_layer_weight(local_layer_weight);
		if (init_met == 1) update_group_weight(start_group);
		//debug
		//best_time = std::chrono::steady_clock::now();//clock();
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &best_time1);
		the_best_time = (double)((best_time1.tv_sec - start_time1.tv_sec) * 1000000 + (double)(best_time1.tv_nsec - start_time1.tv_nsec) / 1000) / 1000000;
		best_found_iter = cur_iter;
		if (is_update_global(best_size)) {
			update_global(best_size, the_best_time, best_plex);
		}
	}

	
	void search_frequency_init() {
		vector<int> vec_M1;
		int leastfreq = LARGE_INT;
		int vstart;
		for (int i = 0; i < 100; i++){
			int randv = rand_r(&rand_seed) % red_vnum;
			if (freq[randv] < leastfreq){
				vec_M1.clear();
				vec_M1.push_back(randv);
				leastfreq = freq[randv];
			}else if (freq[randv] == leastfreq){
				vec_M1.push_back(randv);
			}
		}
		if (vec_M1.size() == 0) vstart = rand_r(&rand_seed) % red_vnum;
		else vstart = vec_M1[rand_r(&rand_seed)%vec_M1.size()];
		//printf("search_frequency_init start v %d red_vnum %d\n", vstart, red_vnum);
		add_cur_vertex(vstart);
		while (1){
			int sat_size = get_saturate_size();
			vec_M1.clear();
			leastfreq = LARGE_INT;
			for (int i = 0; i < cur_cand->vnum; i++){
				int vi = cur_cand->vlist[i];
				int satcon = 0;
				if (cur_c_deg[vi] >= cur_splex->vnum - param_s + 1){
					//verify the connection to staturation vertices
					for (int idx = 0; idx < red_v_edge_cnt[vi]; idx++){
						int vcur = red_v_adj_vertex[vi][idx];
						if (is_in_c[vcur] && Is_Saturated(vcur))
							satcon++;
					}
					if (satcon == sat_size){
						if (freq[vi] < leastfreq){
							vec_M1.clear();
							vec_M1.push_back(vi);
							leastfreq = freq[vi];
						}else if(freq[vi] == leastfreq){
							vec_M1.push_back(vi);
						}
					}
				}
			}
			if (vec_M1.empty()){
				break;
			}
			iter_step++;
			int vadd = vec_M1[rand_r(&rand_seed)%vec_M1.size()];
			//printf("search_frequency_init v %d red_vnum %d\n", vadd, red_vnum);
			add_cur_vertex(vadd);
		}
		init_met = 2;
	}

	void search_group_init(){
		int satcon, sat_size, vadd;
        vector<int> vec_M1;

		pair<int, int> start = get_start_vertex();
		start_group = start.first;
		//printf("search_group_init start v %d red_vnum %d\n", start.second, red_vnum);
		add_cur_vertex(start.second);
		while (1){
			sat_size = get_saturate_size();
			vec_M1.clear();
			int leastfreq = LARGE_INT;
			//memset(mark, 0, sizeof(int) * red_vnum);
			for (int i = 0; i < cur_cand->vnum; i++){
				int vi = cur_cand->vlist[i];
				if (!is_v_in_group(vi)) continue;
				satcon = 0;
				if (cur_c_deg[vi] >= cur_splex->vnum - param_s + 1){
					//verify the connection to staturation vertices
					for (int idx = 0; idx < red_v_edge_cnt[vi]; idx++){
						int vcur = red_v_adj_vertex[vi][idx];
						if (is_in_c[vcur] && Is_Saturated(vcur))
							satcon++;
					}
					if (satcon == sat_size){
						if (freq[vi] < leastfreq){
							vec_M1.clear();
							vec_M1.push_back(vi);
							leastfreq = freq[vi];
						}else if(freq[vi] == leastfreq){
							vec_M1.push_back(vi);
						}
						//vec_M1.push_back(vi);
						//mark[vi] = 1;
					}
				}
			}
			if (vec_M1.empty()){
				break;
			}
			iter_step++;
			vadd = vec_M1[rand_r(&rand_seed)%vec_M1.size()]; //select_add_vertex (vec_M1, mark);
			//printf("search_group_init v %d red_vnum %d\n", vadd, red_vnum);
			add_cur_vertex(vadd);
		}
		init_met = 1;
	}

	void search_greedy_init (){
		vector<int> vec_M1;
		int leastfreq = -99999999;
		memset(profit, 0, red_vnum * sizeof(int));
		for (int i = 0; i < 50; i++) {
			int randv = rand_r(&rand_seed) % red_vnum;
			if (red_v_edge_cnt[randv] > leastfreq) {
				vec_M1.clear();
				vec_M1.push_back(randv);
				leastfreq = red_v_edge_cnt[randv];
			}
			else if (red_v_edge_cnt[randv] == leastfreq) {
				vec_M1.push_back(randv);
			}
		}
		int vstart = vec_M1[rand_r(&rand_seed) % vec_M1.size()];
		add_cur_vertex(vstart);
		while (1) {
			int sat_size = get_saturate_size();
			vec_M1.clear();
			leastfreq = -99999999;
			//print_current_contex();
			for (int i = 0; i < cur_cand->vnum; i++) {
				int vi = cur_cand->vlist[i];
				int satcon = 0;
				if (cur_c_deg[vi] >= cur_splex->vnum - param_s + 1) {
					//verify the connection to staturation vertices
					for (int idx = 0; idx < red_v_edge_cnt[vi]; idx++) {
						int vcur = red_v_adj_vertex[vi][idx];
						if (is_in_c[vcur] && Is_Saturated(vcur))
							satcon++;
					}
					if (satcon == sat_size) {
						if (profit[vi] > leastfreq) {
							vec_M1.clear();
							vec_M1.push_back(vi);
							leastfreq = profit[vi];
						}
						else if (profit[vi] == leastfreq) {
							vec_M1.push_back(vi);
						}
					}
				}
			}
			if (vec_M1.empty()) {
				break;
			}
			int vadd = vec_M1[rand_r(&rand_seed) % vec_M1.size()];
			add_cur_vertex(vadd);
			//print_current_contex();
		}
	}

	
	void fast_init_solution(int thread_id){
		
		if((!is_group_empty()) && (rand_r(&rand_seed)%100 < 60)) {
			search_group_init();
		}
		else {
			search_frequency_init(); 
		}
		
		
		while( cur_splex->vnum < param_s){
			if (cur_splex->vnum == red_vnum)
			{
				clock_gettime(CLOCK_THREAD_CPUTIME_ID, &best_time1);
				totime = (double)((best_time1.tv_sec - start_time1.tv_sec) * 1000000 + (double)(best_time1.tv_nsec - start_time1.tv_nsec) / 1000) / 1000000;
				
				break;
			}
			int vrand = rand_r(&rand_seed) % red_vnum;
			while (is_in_c[vrand]) vrand = rand_r(&rand_seed) % red_vnum;
			add_cur_vertex(vrand);
		}
		if (cur_splex->vnum > best_size){
			record_best();
		}
		
	}

    int select_add_vertex (vector<int>& vec_cand, int *mark) {
		int count1, count2, max_count = 0;
		int best_v = -1;
		int t = 100;
		if (vec_cand.size() > t) {
			for (int i = 0; i < t; i++) {
				int v = vec_cand[rand_r(&rand_seed) % vec_cand.size()];
				count1 = count2 = 0;
				for (int j = 0; j < red_v_edge_cnt[v]; j++) {
					int adjv = red_v_adj_vertex[v][j];
                	if (mark[adjv] == 1) count1++;
					if ((mark[adjv] == 0) && (cur_c_deg[adjv] + 1 >= cur_splex->vnum - param_s + 1)) {
						//printf("deg %d cursize %d k %d\n", cur_c_deg[adjv], cur_splex->vnum, param_s);
						count2++;
					}
				}
				if (count1 > max_count) {
					max_count = count1;
					best_v = v;
				}
			}
		}
		else {
			for (int i = 0; i < vec_cand.size(); i++) {
				int v = vec_cand[i];
				count1 = count2 = 0;
				for (int j = 0; j < red_v_edge_cnt[v]; j++) {
					int adjv = red_v_adj_vertex[v][j];
                	if (mark[adjv] == 1) count1++;
					if ((mark[adjv] == 0) && (cur_c_deg[adjv] + 1 >= cur_splex->vnum - param_s + 1)) {
						//printf("deg %d cursize %d k %d\n", cur_c_deg[adjv], cur_splex->vnum, param_s);
						count2++;
					}
				}
				if (count1 > max_count) {
					max_count = count1;
					best_v = v;
				}
			}
		}
		
		if (cur_splex->vnum + 1 + max_count <= best_size) best_v = -1;
		return best_v;
	}


	/*int select_add_vertex (vector<int>& vec_cand, int *mark) {
		int count1, count2, best_v = -1;
		float score, max_score = 0;
		vector<int> vec_temp;
		int t = 100;
		if (vec_cand.size() > t) {
			for (int i = 0; i < t; i++) {
				int v = vec_cand[rand_r(&rand_seed) % vec_cand.size()];
				count1 = count2 = 0;
				for (int j = 0; j < red_v_edge_cnt[v]; j++) {
					int adjv = red_v_adj_vertex[v][j];
                	if (mark[adjv] == 1) count1++;
					if ((mark[adjv] == 0) && (cur_c_deg[adjv] + 1 >= cur_splex->vnum - param_s + 1)) {
						//printf("deg %d cursize %d k %d\n", cur_c_deg[adjv], cur_splex->vnum, param_s);
						count2++;
					}
				}
				score = cur_c_deg[v] * 0.6 + (count1 + count2) * 0.4;
				//printf("v %d inde %d count1 %d count2 %d score %f\n", v, cur_c_deg[v], count1, count2, score);
				//if (count1 + count2 > max_score) {
			    if (score > max_score) {
					max_score = score; //count1 + count2;
					vec_temp.clear();
					vec_temp.push_back(v);
				}
				//else if (count1 + count2 == max_score) {
				else if (score == max_score) {
					vec_temp.push_back(v);
				}
			}
		}
		else {
			for (int i = 0; i < vec_cand.size(); i++) {
				int v = vec_cand[i];
				count1 = count2 = 0;
				for (int j = 0; j < red_v_edge_cnt[v]; j++) {
					int adjv = red_v_adj_vertex[v][j];
                	if (mark[adjv] == 1) count1++;
					if ((mark[adjv] == 0) && (cur_c_deg[adjv] + 1 >= cur_splex->vnum - param_s + 1)) {
						//printf("deg %d cursize %d k %d\n", cur_c_deg[adjv], cur_splex->vnum, param_s);
						count2++;
					}
				}
				score = cur_c_deg[v] * 0.6 + (count1 + count2) * 0.4;
				//printf("v %d inde %d count1 %d count2 %d score %f\n", v, cur_c_deg[v], count1, count2, score);
				//if (count1 + count2 > max_score) {
			    if (score > max_score) {
					max_score = score; //count1 + count2;
					vec_temp.clear();
					vec_temp.push_back(v);
				}
				//else if (count1 + count2 == max_score) {
				else if (score == max_score) {
					vec_temp.push_back(v);
				}
			}
		}
		
		if (vec_temp.empty()) best_v = vec_cand[rand_r(&rand_seed) % vec_cand.size()];
		else best_v = vec_temp[rand_r(&rand_seed) % vec_temp.size()];
		// for (int i = 0; i < vec_cand.size(); i++) printf("%d(%d %d) ", vec_cand[i], v_in_layer[vec_cand[i]], cur_c_deg[vec_cand[i]]);
		// printf("\n");
		//printf("select cur_splex->vnum %d max_score %f best_v %d best_size %d vec_cand %ld\n", cur_splex->vnum, max_score, best_v, best_size, vec_cand.size());
		
		return best_v;
	}*/


	
	/*find the only unadjacent saturated vertex of v*/
	int find_unadj_satu(int v){
		assert(!is_in_c[v]);
		int v_rt = -1;
		memset(mark, 0, sizeof(int) * cur_splex->vnum);
		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int vcur = red_v_adj_vertex[v][i];
			if (is_in_c[vcur]){
				mark[cur_splex->vpos[vcur]] = 1;
			}
		}
		for (int i = 0; i < cur_splex->vnum; i++){
			int vin = cur_splex->vlist[i];
			//printf("i %d find %d %d %d\n", i, Is_Saturated(vin), mark[i], Is_Saturated(vin) && mark[i] == 0);
			if (Is_Saturated(vin) && mark[i] == 0){
				v_rt = vin;
				//printf("1 sat v %d\n", v_rt);
				break;
			}
			
		}
		//printf("v %d cnt %d cur splex %d v_rt %d\n", v, red_v_edge_cnt[v], cur_splex->vnum, v_rt);
		return v_rt;
	}
	
	/*get a random vertex of C\N_C(v)*/
	int random_unadj_with_exception(int v, int vexception){
		assert(!is_in_c[v]);
		vector<int> vec_unadj;
		memset(mark, 0, sizeof(int) * cur_splex->vnum);
		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int vcur = red_v_adj_vertex[v][i];
			if (is_in_c[vcur]){
				mark[cur_splex->vpos[vcur]] = 1;
			}
		}
		for (int i = 0; i < cur_splex->vnum; i++){
			//printf("i %d random %d %d %d\n", i, !mark[i], cur_splex->vlist[i], !mark[i] && cur_splex->vlist[i] != vexception);
			if (!mark[i] && cur_splex->vlist[i] != vexception)
				vec_unadj.push_back(cur_splex->vlist[i]);
		}
		//printf("vec_unadj.size %ld\n", vec_unadj.size());
		if (vec_unadj.size() == 0)
			return -1;
		else
			return vec_unadj[rand_r(&rand_seed) % vec_unadj.size()];
	}
	
	void push_vertex_tabu(int vpush){
		add_cur_vertex(vpush);
		/*repair*/
		int idx = 0;
		while (idx < cur_splex->vnum){
			int vin = cur_splex->vlist[idx];
			if (Is_Overflow(vin)){
				remove_cur_vertex(vin);
				tabu_add[vin] = cur_iter; // + 4;
				//ATTENTION: idx need to be hold for another round
				//freq[vin]++;
			}else
				idx++;
		}
	}

	void push_vertex_tabu1(int vpush){
		/*repair*/
		int idx = 0;
		while (cur_c_deg[vpush] <= cur_splex->vnum - param_s){
			int vin = cur_splex->vlist[rand_r(&rand_seed) % cur_splex->vnum];
			remove_cur_vertex(vin);
			tabu_add[vin] = cur_iter; // + 4;
		}
		add_cur_vertex(vpush);
	}
	
	int get_most_momentum(vector<int>& perturb_set) {
		double min_momentum = -999999999;
		vector<int> cand;
		for (int i = 0, size = perturb_set.size(); i < size; ++i) {
			double curr_score = (double)cur_c_deg[perturb_set[i]] + 0.2 * ((double)cur_c_deg[perturb_set[i]] / red_vnum)*(red_v_edge_cnt[perturb_set[i]] - cur_c_deg[perturb_set[i]]);
			if (curr_score > min_momentum) {
				min_momentum = cur_c_deg[perturb_set[i]];
				cand.clear();
				cand.push_back(perturb_set[i]);
			}
			else if (curr_score == min_momentum) {
				cand.push_back(perturb_set[i]);
			}
		}
		return cand[rand_r(&rand_seed) % cand.size()];
	}
	
	pair<int, int> get_most_momentum_2(vector<pair<int, int> > perturb_set) {
		int min_momentum = 999999999;
		vector<pair<int, int> > cand;
		int best_subscore = -1;
		for (int i = 0, size = perturb_set.size(); i < size; ++i) {
			if (freq[perturb_set[i].first] < min_momentum) {
				min_momentum = freq[perturb_set[i].first];
				cand.clear();
				cand.push_back(perturb_set[i]);
			}
			else if (freq[perturb_set[i].first] == min_momentum) {
				cand.push_back(perturb_set[i]);
			}
		}
		return cand[rand_r(&rand_seed) % cand.size()];
	}
	
	void tabu_based_search(int thread_id){
		vector<int> vec_M1;
		vector<pair<int,int>> vec_M2_out;
		vector<int> vec_M3;
		int cycle_iter = 0;
		int max_cycle_iter; //= 4000; //param_cycle_iter;
		if (org_vnum >= 1000000) max_cycle_iter = 2000;
		else max_cycle_iter = 1000;
		int cycle_best = cur_splex->vnum;
		int fixed = -1;
		int max_freq;
		int non_improve_iter = 0;
		int v_remove;
		while (1){
			int end = 0;
			int core_perturb_flag = 0;
			vector<int> core_set;
			while (!end){
				/*           */
				vector<int> saturated_vector;
				int satu_size = 0;
				for (int i = 0; i < cur_splex->vnum; i++) {
					int vin = cur_splex->vlist[i];
					if (Is_Saturated(vin)) {
						satu_size++;
						if (core_perturb_flag == 1) {
							saturated_vector.push_back(vin);
						}
					}
				}
				
				if (core_perturb_flag == 1) {
					if (satu_size == 0) {
						core_perturb_flag = 2;
						continue;
					}
					int min_freq_ver = saturated_vector[0];
					int min_freq = tabu_add[saturated_vector[0]];
					for (int i = 1; i < saturated_vector.size(); i++) {
						if (tabu_add[saturated_vector[i]] < min_freq) {
							min_freq = tabu_add[saturated_vector[i]];
							min_freq_ver = saturated_vector[i];
						}
					}
					ver_inc_deg0.clear();
					remove_cur_vertex(min_freq_ver);//移除freq大的点，保留freq小的点
					core_set.push_back(min_freq_ver);
					tabu_add[min_freq_ver] = cur_iter;
					continue;
				}
				if (core_perturb_flag == 2) {
					for (int i = 0; i < core_set.size() / 2; i++) {
						int min_freq = INT_MAX;
						int min_freq_ver = -1;
						for (int j = 0; j < core_set.size(); j++) {
							if (is_in_c[core_set[j]] == 1)
								continue;
							if (min_freq > freq[core_set[i]]) {
								min_freq_ver = core_set[i];
								min_freq = freq[core_set[i]];
							}
						}
						assert(min_freq_ver != -1);
						add_cur_vertex(min_freq_ver);//假如freq小的点
						tabu_add[min_freq_ver] = cur_iter;
						
					}
					core_perturb_flag = 0;
					core_set.clear();
					continue;
				}
				
				vec_M1.clear();
				vec_M2_out.clear();
				vec_M3.clear();
				max_freq = 0;
				
				/*Update M1 M2 sets*/
				for (int i = 0; i < cur_cand->vnum; i++){
					int vcand = cur_cand->vlist[i];
					int satcon = 0;
					if (cur_c_deg[vcand] >= cur_splex->vnum - param_s){
						//check all the adjacent vertices of vcan in C
						for (int idx = 0; idx < red_v_edge_cnt[vcand]; idx++){
							int vcur = red_v_adj_vertex[vcand][idx];
							if (is_in_c[vcur] && Is_Saturated(vcur)){
								satcon++;
							}
						}
						if (cur_c_deg[vcand] >= cur_splex->vnum - param_s + 1
								&& satcon == satu_size
								&& (deposit[vcand] >= threshold[vcand] || cycle_iter - tabu_add[vcand] > 3 || cur_splex->vnum + 1 > best_size)){
							vec_M1.push_back(vcand);
						}else if (cur_c_deg[vcand] >= cur_splex->vnum - param_s
								&& satcon == satu_size - 1
								&& (deposit[vcand] >= threshold[vcand] || cycle_iter - tabu_add[vcand] > 5)){
							vec_M2_out.push_back(make_pair(vcand, 0));
						}else if (cur_c_deg[vcand] == cur_splex->vnum - param_s
								&& satcon == satu_size
								&& (deposit[vcand] >= threshold[vcand])){
							vec_M2_out.push_back(make_pair(vcand, 1));
						}else if (deposit[vcand] >= threshold[vcand] || cycle_iter - tabu_add[vcand] > 7 ){
							if (freq[vcand] > max_freq){
								vec_M3.clear();
								vec_M3.push_back(vcand);
								max_freq = vcand;
							}else if (freq[vcand] == max_freq){
								vec_M3.push_back(vcand);
							}
						}
					}else if (deposit[vcand] >= threshold[vcand] || cycle_iter - tabu_add[vcand] > 7 ){
						if (freq[vcand] > max_freq){
							vec_M3.clear();
							vec_M3.push_back(vcand);
							max_freq = vcand;
						}else if (freq[vcand] == max_freq){
							vec_M3.push_back(vcand);
						}
					}
				}
	            /*Add or Swith move*/
				if (!vec_M1.empty()){
					int vadd = get_most_momentum(vec_M1);
					add_cur_vertex(vadd);
					if (cur_splex->vnum > cycle_best){
						cycle_best = cur_splex->vnum;
					}
					tabu_add[vadd] = cur_iter;
					
				}
				else if (!vec_M2_out.empty()){
					pair<int, int> pvswp = get_most_momentum_2(vec_M2_out);
					int vswp_in;
					if (pvswp.second == 0){ //type 1
						vswp_in = find_unadj_satu(pvswp.first);
					}else{ //type 2
						vswp_in = random_unadj_with_exception(pvswp.first, fixed);
					}
					if (vswp_in == fixed || vswp_in == -1) end = 1;
					else {
						remove_cur_vertex(vswp_in);
						tabu_add[vswp_in] = cur_iter;
						add_without_change_deposit(pvswp.first);
						tabu_add[pvswp.first] = cur_iter;
					}
					//freq[vswp_in]++;
					//freq[pvswp.first]++;
				}
				else{
					end = 1;
				}
				if (cur_splex->vnum > best_size){
					record_best();
					if (best_size == param_best) //reach optimum
						goto ts_stop;
					non_improve_iter = 0;
				}
				else{
					non_improve_iter++;
				}
				if (non_improve_iter > min(param_s * best_size2, 500)){
					non_improve_iter = 0;
					end = 1;
					core_perturb_flag = 1;
				}
				/*if (non_improve_iter > param_s * best_size){
					non_improve_iter = 0;
					end = 1;
				}*/
				cur_iter++;
				cycle_iter++;
				iter_step++;
                clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time1);
				totime = (double)((total_time1.tv_sec - start_time1.tv_sec) * 1000000 + (double)(total_time1.tv_nsec - start_time1.tv_nsec) / 1000) / 1000000;
				if(cycle_iter > max_cycle_iter || totime > param_max_seconds){
					if(updated.load()) {
						max_cycle_iter += 100;
						continue;
					}
					else goto ts_stop;
			    }

				
			}
			if (cur_splex->vnum == 0) {
				goto ts_stop;
			}
			if (!vec_M3.empty()) {
				v_remove = vec_M3[rand_r(&rand_seed) % vec_M3.size()];
				fixed = v_remove;
				push_vertex_tabu(v_remove);
				tabu_add[v_remove] = cur_iter;
			}
			else {
				
				
				if ((cur_remain->vnum > 0) && (rand_r(&rand_seed)%100 < 5)) {
					int vrand = cur_remain->vlist[rand_r(&rand_seed) % cur_remain->vnum];
				    fixed = vrand;
					push_vertex_tabu1(vrand);
				    tabu_add[vrand] = cur_iter;
				}
				else {
					break;
				}

			}
			
			
		}
		
	ts_stop:
		return;
	}

	void update_step_in_solution() {
		for (int i = 0; i < cur_splex->vnum; i++) {
			int v = cur_splex->vlist[i];
			step_in_solution[v] += iter_step - add_step[v]; 
		}
	}

	void update_red(int x, int thread_id)
	{
		while(org_decompos[red_orgid[red_vnum - 1]] <= x)
		{
			red_vnum--;
			//delete[] red_v_adj_vertex[red_vnum];
			if(red_vnum == 0) return;
		}
		
		red_min_deg = org_decompos[red_orgid[red_vnum - 1]];
		for(int i = 0;i < red_vnum;i++)
		{
			while(org_decompos[red_orgid[red_v_adj_vertex[i][red_v_edge_cnt[i] - 1]]] <= x)
			{
				red_v_edge_cnt[i] --;
				if(red_v_edge_cnt[i] == 0) break;
			}
		}
		
		if (is_update_group(red_vnum) && (!is_group_empty())) {
			vector<int> vec_M1;
			for (int i = 0; i < best_size; i++) {
				int v = new_id[best_plex[i]];
				vec_M1.push_back(v);
			}
			update_group(red_vnum, thread_id, vec_M1);
		}
	}

	void search_mode1 (int thread_id) {
		int reduce_num;

		/*start from a new solution*/
		fast_init_solution(thread_id);
		if (cur_splex->vnum == red_vnum) return;
		reduce_num = is_reduced_group(red_min_deg);
		if (reduce_num > 0) {
			update_red(reduce_num, thread_id);
			
			if (red_vnum == 0) return ;
		}
		

		/*start for search a better solution*/
		while (1){ 
			restart_search();
			fast_init_solution(thread_id);
			if (cur_splex->vnum == red_vnum) return;
			
			tabu_based_search(thread_id);
			
			update_step_in_solution();
			
			reduce_num = is_reduced_group(red_min_deg);
			
			if (reduce_num > 0) {
				update_red(reduce_num, thread_id);
				
				if ((best_size == param_best) || (red_vnum <= best_size)) {
					clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time1);
					totime = (double)((total_time1.tv_sec - start_time1.tv_sec) * 1000000 + (double)(total_time1.tv_nsec - start_time1.tv_nsec) / 1000) / 1000000;
					
					return;
				}
			}
				
			restart_pass++;
			whether_update_group(thread_id);


			clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time1);
			totime = (double)((total_time1.tv_sec - start_time1.tv_sec) * 1000000 + (double)(total_time1.tv_nsec - start_time1.tv_nsec) / 1000) / 1000000;
			if (totime > param_max_seconds ){
				break;
			}

		}
		ral_release(cur_splex);
		ral_release(cur_cand);
		ral_release(cur_remain);
	}

	void search_mode2 (int thread_id) {
		int v, startv, index, startindex ,sat_size, reduce_num;
		int end = 0, step = 0;;
		vector<int> start_vec;
		vector<int> vec_M1;
    
        while(!end) {
			
			restart_search();
			step ++;
			startv = select_start_vertex(freq, red_vnum);
			add_cur_vertex(startv);
			
			while(1) {
				sat_size = get_saturate_size();
				vec_M1.clear();
				memset(mark, 0, sizeof(int) * red_vnum);
				for (int i = 0; i < cur_cand->vnum; i++) {
					v = cur_cand->vlist[i];
					int satcon = 0;
					if (cur_c_deg[v] >= cur_splex->vnum - param_s + 1) {
						for (int idx = 0; idx < red_v_edge_cnt[v]; idx++) {
							int vcur = red_v_adj_vertex[v][idx];
							if (is_in_c[vcur] && Is_Saturated(vcur)) satcon++;
						}
						if (satcon == sat_size) {
							vec_M1.push_back(v);
							mark[v] = 1;
							
						}
					}
				}
                if (vec_M1.empty()) {
					
					break;
				}
				v = select_add_vertex (vec_M1, mark);
				if (v == -1) break;
				add_cur_vertex(v);
				iter_step ++;

				clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time1);
				totime = (double)((total_time1.tv_sec - start_time1.tv_sec) * 1000000 + (double)(total_time1.tv_nsec - start_time1.tv_nsec) / 1000) / 1000000;
				if (totime > param_max_seconds ){
					end = 1;
					break;
				}
			}
			

			if (cur_splex->vnum > best_size){
				record_best();
			}
			update_step_in_solution();
			
			reduce_num = is_reduced_group(red_min_deg); //best_size - param_s;   
			if (reduce_num > 0) {
				update_red(reduce_num, thread_id);
				if ((best_size == param_best) || (red_vnum <= best_size)) {
			        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time1);
					totime = (double)((total_time1.tv_sec - start_time1.tv_sec) * 1000000 + (double)(total_time1.tv_nsec - start_time1.tv_nsec) / 1000) / 1000000;
					return;
				}
			}
			

			restart_pass++;
			whether_update_group(thread_id);
			clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time1);
			totime = (double)((total_time1.tv_sec - start_time1.tv_sec) * 1000000 + (double)(total_time1.tv_nsec - start_time1.tv_nsec) / 1000) / 1000000;
			if (totime > param_max_seconds ){
				break;
			}
		}
		ral_release(cur_splex);
		ral_release(cur_cand);
		ral_release(cur_remain);
	}
	
	
	/*TODO:Entrance of the whole search*/
	void search_main(int thread_id){
		
		/*Initial data structure*/
		init_search(thread_id);
		
		
		if (thread_id == 9) search_mode2(thread_id);
		else search_mode1 (thread_id);
	
		//total_time = std::chrono::steady_clock::now();//clock();
		//clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time1);
	}

	void report_result(){
		cout << param_s << " " << param_graph_file_name << " " << param_seed << " " << best_size << " " << " " << the_best_time << " " << rand_seed_record << " " << org_vnum << " " << org_enum << " " << 2*org_enum/((org_vnum -1)*org_vnum*1.0) << " " << red_vnum << endl;
	    //for (int i = 0; i < red_vnum; i++) cout << i << " ";
		//cout << endl;
		//cout << param_s << param_graph_file_name << " " << param_seed << " " << best_size << " " << std::chrono::duration<double>(best_time - start_time).count() << " " << rand_seed_record << " " << init_search_step << " " << init_search_time / (double)init_search_step << " " << init_search_time1 / (double)init_search_step << " " << fast_inso_step <<" " << fast_inso_time / (double)fast_inso_step << " " << fast_inso_time1 / (double) fast_inso_step << " " << update_red_step << " " << update_red_time / (double) update_red_step << " " << update_red_time1 / (double)update_red_step <<" " << res_search_step << " " << res_search_time / (double) res_search_step << " " << res_search_time1 / (double) res_search_step <<" " << tabu_search_step << " " << tabu_search_time / (double)tabu_search_step << " " << tabu_search_time1 / (double)tabu_search_step << test_step1 << " " << test_time1/(double)test_step1 << " " << test_step2 << " " << test_time2/(double)test_step2 << " " << test_step3 << " " << test_time3/(double)test_step3 << " " << test_step4 << " " << test_time4/(double)test_step4 <<" " << test_step5 << " " << test_time5/(double) test_step5 << " " << test_step6 << " " << test_time6/(double)test_step6 << " " << test_step7 << " " << test_time7/(double)test_step7 << " " << test_step8 <<" " << test_time8/(double)test_step8 << endl;
		//cout << param_s << param_graph_file_name << " " << param_seed << " " << best_size << " " << std::chrono::duration<double>(best_time - start_time).count() << " " << rand_seed_record << " " << res_search_step << " " << res_search_time / (double) res_search_step << " " << res_search_time1 / (double) res_search_step <<" " << tabu_search_step << " " << tabu_search_time / (double)tabu_search_step << " " << tabu_search_time1 / (double)tabu_search_step << " " << test_step1 << " " << test_time1/(double)test_step1 << " " << test_step2 << " " << test_time2/(double)test_step2 << " " << test_step3 << " " << test_time3/(double)test_step3 << " " << test_step4 << " " << test_time4/(double)test_step4 <<" " << test_step5 << " " << test_time5/(double) test_step5 << " " << test_step6 << " " << test_time6/(double)test_step6 << " " << test_step7 << " " << test_time7/(double)test_step7 << " " << test_step8 <<" " << test_time8/(double)test_step8 << " " << tabu_search_time << " " << test_time1 + test_time2 + test_time3 + test_time4 + test_time5 + test_time6 + test_time7 + test_time8 << endl;
	
	}
	
	
	
	int check_solution(){
		int *mark = new int[org_vnum];
		memset(mark, 0, sizeof(int) * org_vnum);
		for (int i = 0; i < best_size; i++){
			mark[best_plex[i]] = 1;
		}
		//printf("best size %d\n", best_size);
		for (int i = 0; i < best_size; i++){
			int v = best_plex[i];
			int indeg = 0;
			//printf("%d:", new_id[v]); //v_in_layer[new_id[v]]);
			for (int j = 0; j < org_v_edge_cnt[v]; j++){
				int vadj = org_v_adj_vertex[v][j];
				if (mark[vadj]) {
					indeg++;
					//printf("%d ", new_id[vadj]);
				}	
			}
			//printf("\n");
			//printf("(%d) ", indeg);
			if (indeg < best_size - param_s)
				return 0;
		}
		delete[] mark;
		return 1;
	}
	
	/*const char* file_suffix(char* filename){
	    const char *dot = strrchr(filename, '.');
	    if(!dot || dot == filename) return "";
	    return dot + 1;
	}*/
}; 

bool cmp(int a,int b)
{
	return org_decompos[red_orgid[a]] > org_decompos[red_orgid[b]];
}

/*void sortList(int *tmp_lst,int size)
{
	auto cmpfunc = [this](int a, int b) {
        return this->cmp(a, b);
    };
    sort(tmp_lst, tmp_lst + size, cmpfunc);
}*/

Splex* splex_solver = nullptr;


set<st> s1;

bool cmp2(int a,int b)
{
	return a > b;
}

void global_init_search()
{
	min_vnum = org_vnum;
	red_orgid = new int[org_vnum];
	red_v_adj_vertex = new int*[org_vnum];
	int *degree;
	degree = new int[org_vnum];
	k_decompos = new int*[org_vnum];
	k_cnt_decompos = new int[org_vnum];
	org_decompos = new int[org_vnum];
	layers = new int[org_vnum];
	sum_layers = new long long[org_vnum];
	v_in_layer = new int[org_vnum];
	global_best_plex = new int[org_vnum];
	cnt_layers = 0;
	for (int v = 0; v < org_vnum; v++) {
		red_orgid[v] = v;
		red_v_adj_vertex[v] = new int[org_v_edge_cnt[v]];
		memcpy(red_v_adj_vertex[v], org_v_adj_vertex[v], sizeof(int) * org_v_edge_cnt[v]);
		degree[v] = org_v_edge_cnt[v];
		
	}

	
	bool *vis;
	vis = new bool[org_vnum];
	s1.clear(); 
	for(int i = 0;i < org_vnum;i++) 
	{
		//printf("v %d degree %d\n", i+1, degree[i]);
		s1.insert((st){i,degree[i]});
		vis[i] = 0; 
	}
	/*int ccount = 0;
	set<st>::iterator iter;
	for(iter = s1.begin();iter != s1.end();++iter)
	{
		printf("v %d degree %d\n", (*iter).n, (*iter).x);
		ccount ++;
	}
	printf("some %d\n", ccount);*/
    //printf("bbbbbbbbb %d %d\n", (*(s1.begin())).n, (*(s1.begin())).x);
		queue<int> q;
		int min_degree = 2147483647;
		int *k_temporary;
		k_temporary = new int[org_vnum];
		int k_cnt_temporary = 0;
		while(!s1.empty())
		{
			stack<st> sta;
			int min_degree = (*(s1.begin())).x;
			//printf("vertex %d min_dgree %d\n", (*(s1.begin())).n, (*(s1.begin())).x);
			set<st>::iterator it;
			for(it = s1.begin();it != s1.end();++it)
			{
				//printf("v %d degree %d\n", (*it).n, (*it).x);
				if((*it).x <= min_degree) 
				{
					q.push((*it).n);
					vis[(*it).n] = 1;
					sta.push(*it);
				}
				else break;
			}
			while(!sta.empty())
			{
				s1.erase(sta.top());
				sta.pop();
			}
			
			layers[cnt_layers++] = min_degree;
			//printf("cnt_layers:%d min_degree:%d\n", cnt_layers, min_degree);
			while(!q.empty())
			{
				int u = q.front();
				q.pop();
				k_temporary[k_cnt_temporary++] = u;
				org_decompos[u] = min_degree;
				for(int i = 0;i < org_v_edge_cnt[u];i++)
				{
					int v = red_v_adj_vertex[u][i];
					degree[u]--;
					degree[v]--;
					if(vis[v]) continue;
					if(degree[v] <= min_degree)
					{
						s1.erase((st){v,degree[v] + 1});   // ?
						q.push(v);
						vis[v] = 1;
					}
					else
					{
						s1.erase((st){v,degree[v] + 1}); 
						s1.insert((st){v,degree[v]});
					}
				}
			}
			k_cnt_decompos[min_degree] = k_cnt_temporary;	
			
			k_decompos[min_degree] = new int[k_cnt_temporary];
			memcpy(k_decompos[min_degree], k_temporary, sizeof(int) * k_cnt_temporary);
			k_cnt_temporary = 0;
		}
		int *temp = new int[org_vnum];
		
		for(int i = 0;i < org_vnum;i++) temp[i] = i;
		sort(temp,temp + org_vnum,cmp);  //  按顶点对应的最小度从大到小排序
		new_id = new int[org_vnum];
		for(int i = 0;i < org_vnum;i++) new_id[temp[i]] = i; // 按排序重新编号
		red_orgid = temp;
		new_adj_tbl = new int*[org_vnum];
		new_edge_count = new int[org_vnum];
		
		for (int idx_prev = 0; idx_prev < org_vnum; idx_prev++)
		{
			int idx_new = new_id[idx_prev];
			new_adj_tbl[idx_new] = new int[org_v_edge_cnt[idx_prev]];
			new_edge_count[idx_new] = org_v_edge_cnt[idx_prev];  // 把原来的顶点度赋给新编号对应的顶点度
			int cnt = 0;
			for (int i = 0; i < org_v_edge_cnt[idx_prev]; i++) {
				int vi_adj = red_v_adj_vertex[idx_prev][i];
				new_adj_tbl[idx_new][cnt++] = new_id[vi_adj];
			}
			
			sort(new_adj_tbl[idx_new],new_adj_tbl[idx_new] + new_edge_count[idx_new],cmp);
			//printf("old %d new %d cnt:%d org_decom %d\n", idx_prev, idx_new, cnt, org_decompos[idx_prev]);
			//for(int i = 0; i < cnt; i++) printf("%d(%d) ", new_adj_tbl[idx_new][i], org_decompos[red_orgid[new_adj_tbl[idx_new][i]]]);
			//printf("\n");
		}

	
		// for (int i = 0; i < org_vnum; i++) {
		// 	printf("new %d old %d org_decom %d\n", i, red_orgid[i], org_decompos[red_orgid[i]]);
		// 	for (int j = 0; j < new_edge_count[i]; j++) printf("%d(%d) ", new_adj_tbl[i][j], org_decompos[red_orgid[new_adj_tbl[i][j]]]);
		// 	printf("\n");
		// }

        avg_mindeg = 0;
        //printf("cnt layer %d org_vnum %d\n", cnt_layers, org_vnum);
		for(int i = 0;i < cnt_layers;i++) {
			//printf("i %d size %d minged %d\n",i, k_cnt_decompos[layers[i]], layers[i]);
			for(int j = 0;j < k_cnt_decompos[layers[i]];j++){
				k_decompos[layers[i]][j] = new_id[k_decompos[layers[i]][j]];
				v_in_layer[k_decompos[layers[i]][j]] = i; //layers[i];
			}
			avg_mindeg = avg_mindeg + layers[i];
		}
		avg_mindeg = avg_mindeg / cnt_layers;
		//printf("avg_mindeg %d\n", avg_mindeg);
		global_layer_weight = new int[cnt_layers];
		memset(global_layer_weight, 0, sizeof(int) * cnt_layers);

        int rem = org_vnum;
        for (int i = 0; i < org_vnum; i++) {
			// printf("v %d: ", i);
			// for (int j = 0; j < new_edge_count[i]; j++) {
			// 	printf("%d ", new_adj_tbl[i][j]);
			// }
			// printf(" (%d)\n", new_edge_count[i]);
			if (new_edge_count[i] == 0) rem --;
	    }
		//printf("rem %d\n", rem);
		if (rem == 0) {
			cout << param_graph_file_name << " is no " << param_s << "-plex." << endl;
			exit(0);
		}
		
		
		//sort(layers, layers + cnt_layers,cmp2);
		//delete[] red_v_adj_vertex;
		red_v_adj_vertex = new_adj_tbl;
		for(int i = 0;i < thread_size;i++) 
		{
			splex_solver[i].red_vnum = rem; //org_vnum;
			splex_solver[i].red_enum = org_enum;
			splex_solver[i].red_v_edge_cnt = new int[org_vnum];
			memcpy(splex_solver[i].red_v_edge_cnt, new_edge_count, sizeof(int) * org_vnum);
		}
		

		// for (int i = 0; i < org_vnum; i++) {
		// 	printf("old %d new %d layer %d\n", i, new_id[i], v_in_layer[new_id[i]]);
		// }
	
	
		sum_layers[0] = layers[0];
	 	for(int i = 1;i <= cnt_layers;i++){
			sum_layers[i] = sum_layers[i - 1] + layers[i - 1];
		}

			

		delete[] degree;
		delete[] vis;
		delete[] k_temporary;
}


void update_v_remain(int vertex, int loc){
	int lastv, tt;
	if (loc == -2) return;
	if (loc != -1) {
		lastv = v_remain.back();
		local_in_remain[lastv] = loc;
		local_in_remain[v_remain[loc]] = -1;
		swap(v_remain[loc], v_remain.back());
		v_remain.pop_back();
		
	}
	if (vertex != -1) {
		tt = v_remain.size();
		local_in_remain[vertex] = tt;
		v_remain.push_back(vertex);
		v_in_tem[vertex] = 0;
	}
		
}


void update_v_cand(vector<int>& vec_cand, int vertex, int loc) {
	int adjv, lastv, tt;
		
	if (!vec_cand.empty()) {
		lastv = vec_cand.back();
		local_in_cand[lastv] = loc;
		local_in_cand[vec_cand[loc]] = -1;
		swap(vec_cand[loc], vec_cand.back());
		vec_cand.pop_back();
	}
	for (int i = 0; i < new_edge_count[vertex]; i++) {
		adjv = new_adj_tbl[vertex][i];
		cur_g_deg[adjv]++;
		if ((local_in_cand[adjv] == -1) && (local_in_remain[adjv] != -1) && (v_in_tem[adjv] == 0)) {
			tt = vec_cand.size();
			local_in_cand[adjv] = tt;
			vec_cand.push_back(adjv);
		}
	}
	v_in_tem[vertex] = 1;
}

void group_partition(){
	int randv, bestv, v, max_layer, max_degree, local_rem, local_can, t, size, sum_v, adjv;
	int iter_try = 0, max_try = 10;
	vector<int> v_cand;
	vector<int> v_temp;
	vector<int> vec_M1;

	cur_g_deg = new int[org_vnum];
	deg_in_group = new int[org_vnum];
	avg_deg_group = new int[org_vnum];
	update_id_group = new int[org_vnum];
	local_in_group = new int[org_vnum];
	local_in_remain = new int[org_vnum];
	local_in_cand = new int[org_vnum];
	v_in_tem = new int[org_vnum];
	group_weight = new int[org_vnum];
	t = 0;
	for (int i = 0; i < org_vnum; i++) {
		if (layers[v_in_layer[i]] > avg_mindeg) {
			v_remain.push_back(i);
			local_in_remain[i] = t++;
		}
		else local_in_remain[i] = -2;
		//printf("v %d minde %d avgde %d local %d\n", i, layers[v_in_layer[i]], avg_mindeg, local_in_remain[i]);
		
		local_in_cand[i] = -1;
		local_in_group[i] = -1;
		v_in_tem[i] = 0;
		deg_in_group[i] = 0;
		avg_deg_group[i] = 0;

		group_weight[i] = 1;
	}
	//printf("t %d\n", t);
        
	group_num = 0;
    v_group.clear();
	while(iter_try < max_try) {
		v_temp.clear();
		memset(cur_g_deg, 0, org_vnum * sizeof(int));
		max_layer = 0;
		max_degree = 0;
		if (v_remain.empty()) break;
		t = v_remain.size() % 100 + 1;
		for (int i = 0; i < t; i++) {
			randv = v_remain[rand() % v_remain.size()];
            if (max_layer < v_in_layer[randv]) {
				v_cand.clear();
				v_cand.push_back(randv);
				max_layer = v_in_layer[randv];
			}
			else if (max_layer == v_in_layer[randv]) {
				v_cand.push_back(randv);
			}
			//printf("v %d degree %d\n", randv, new_edge_count[randv]);
		}
        //printf("max_layer %d  size %ld\n", max_layer, v_cand.size());
			
		
		v = v_cand[rand() % v_cand.size()];
		v_temp.push_back(v);
		local_rem = local_in_remain[v]; 
		//printf("start v %d remain %d cand %d\n", v, local_rem, local_in_cand[v]);
		update_v_remain(-1, local_rem);
		//printf("update remain size %ld v %d local %d lastv %d lastlocal %ld\n", v_remain.size(), v_remain[local_rem], local_rem, v_remain[v_remain.size()-1], v_remain.size()-1);
            
        v_cand.clear();
		update_v_cand(v_cand, v, 0);

        
		while(1){
			size = v_temp.size();
			vec_M1.clear();
			for(int i = 0; i < v_cand.size(); i++) {
				v = v_cand[i];
				if (cur_g_deg[v] > size - param_s ) {
					vec_M1.push_back(v);
				}
			}
			if (vec_M1.empty()) break;
			randv = vec_M1[rand()%vec_M1.size()];
			v_temp.push_back(randv);
			local_rem = local_in_remain[randv];
			local_can = local_in_cand[randv];
			//printf("search v %d remain %d cand %d\n", randv, local_rem, local_can);
			update_v_remain(-1, local_rem);
			update_v_cand(v_cand, randv, local_can);
		}


		if (!v_cand.empty()) {
			for (int i = 0; i < v_cand.size(); i++) {
				local_in_cand[v_cand[i]] = -1;
			}
		}
        
		if (v_temp.size() >= low_group_size) { 
			for (int i = 0; i < v_temp.size(); i++) local_in_group[v_temp[i]] = group_num;
			for (int i = 0; i < v_temp.size(); i++) {
				for (int j = 0; j < new_edge_count[v_temp[i]]; j++) {
					adjv = new_adj_tbl[v_temp[i]][j];
					if (local_in_group[adjv] == group_num) deg_in_group[v_temp[i]]++;
				}
				avg_deg_group[group_num] += deg_in_group[v_temp[i]];
			}
			avg_deg_group[group_num] = avg_deg_group[group_num] / v_temp.size();
			v_group.push_back(v_temp);
			group_num++;
			iter_try = 0;
		}
		else {
			for (int i = 0; i < v_temp.size(); i++) {
				if (local_in_remain[v_temp[i]] == -2) continue;
				update_v_remain(v_temp[i], -1);
			}
			iter_try++;
		}

		//printf("temp size %ld group num %d iter_try %d max_layer %d\n", v_temp.size(), group_num, iter_try, max_layer);
	}
    //printf("init group %ld\n", v_group.size());
	sum_v = 0;
	for (int i = 0; i < v_group.size(); i++) {
		sum_v += v_group[i].size();
		//printf("i %d size %ld weight %d\n", i, v_group[i].size(), group_weight[i]);
	}
	if ((sum_v != 0) && (v_group.size() != 0)) {
		low_group_size = (sum_v / v_group.size()) * gama;
	    up_group_size = (sum_v / v_group.size()) * 2;
	}
    

	//printf("init size %ld  group vnum %d total vnum %d raito %lf\n", v_group.size(), sum_v, org_vnum, (sum_v*1.0)/org_vnum);
    /*for (int i = 0; i < v_group.size(); i++) {
		printf("i %d size %ld\n", i, v_group[i].size());
		for (int j = 0; j < v_group[i].size(); j++) {
			int count = 0;
			for (int l = 0; l < new_edge_count[v_group[i][j]]; l++) {
				adjv = new_adj_tbl[v_group[i][j]][l];
				if (local_in_group[adjv] == i) count ++;
			}
			printf("%d(%d %d) ", v_group[i][j], count, v_in_layer[v_group[i][j]]);
		}
		printf("\n");
	}
	printf("low %d up %d\n", low_group_size, up_group_size);*/

}


void global_free_memory() {
	int i;
	//for (i = 0; i < org_vnum; i++) delete[] k_decompos[i];
	for (i = 0; i < org_vnum; i++) delete[] red_v_adj_vertex[i];
	for (i = 0; i < org_vnum; i++) delete[] org_v_adj_vertex[i];

	//delete[] k_decompos;
	delete[] red_v_adj_vertex;
	delete[] org_v_adj_vertex;
	delete[] org_decompos;
	delete[] k_cnt_decompos;
	delete[] red_orgid;
	delete[] org_vid;
	delete[] org_v_edge_cnt;
	delete[] layers;
	delete[] sum_layers;
	delete[] new_edge_count;
	
	delete[] local_in_cand;
	delete[] local_in_group;
	delete[] local_in_remain;
	delete[] v_in_tem;
	delete[] v_in_layer;	
	delete[] global_layer_weight;
	delete[] cur_g_deg;
	delete[] deg_in_group;
	delete[] avg_deg_group;
	delete[] update_id_group;
	delete[] new_id;
	delete[] group_weight;
	delete[] global_best_plex;
 
}

int isnot_load[1000 + 10];


struct ThreadData {
    int tid;
    int argc;
    char** argv;
}tid_array[100 + 10];

void *create_search(void *arg)
{
	ThreadData *data = (ThreadData*)arg;
	int i = data->tid;
	
	if(isnot_load[i]) return NULL;
	splex_solver[i].search_main(i);
	return NULL;
}

int main(int argc, char** argv) {
	load(argc,argv);
	splex_solver = new Splex[thread_size];
	for(int i = 0;i < thread_size;i++) splex_solver[i] = Splex();
	for(unsigned int i = 0;i < thread_size;i++) splex_solver[i].set_seed(i);
	global_init_search();
	group_partition();
	
	pthread_t *ptr = new pthread_t[thread_size];
	for(int tid = 0;tid < thread_size;tid++) 
	{
		tid_array[tid].tid = tid;
		tid_array[tid].argc = argc;
		tid_array[tid].argv = argv;
		pthread_create(&ptr[tid], NULL, create_search,(void*)&tid_array[tid]);
	}
	for(int tid = 0;tid < thread_size;tid++) pthread_join(ptr[tid], NULL);

	
	gloabl_avg = 0;
	for(int i = 0;i < thread_size;i++)
	{
		int chk = splex_solver[i].check_solution();
		
		if (chk == 0){
			//printf("ERROR! Final solution is infeasible at thread %d\n",i);
			printf("ERROR! %d %s Final solution is infeasible at thread %d\n",param_s,param_graph_file_name,i);
			//exit(0);
		}
		//else splex_solver[i].report_result();

		
		gloabl_avg += splex_solver[i].best_size;
		splex_solver[i].free_memory();
	}
	//cout << param_graph_file_name << " " << param_s << " final " << global_best_size << " " << (double)gloabl_avg / thread_size << " " << global_best_time << endl;
	cout << "instance " << param_graph_file_name << " k " << param_s << " size " << global_best_size << " time " << global_best_time << endl;
	if (global_best_size > 0) {
		cout << "solution:";
		for (int i = 0; i < global_best_size; ++i) {
			cout << " " << global_best_plex[i];
		}
	} else {
		cout << ": <empty>";
	}
	cout << endl;
	global_free_memory();
	//report_result();
}
