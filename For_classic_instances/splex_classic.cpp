#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <libgen.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <algorithm>
#include <sstream>
#include <list>
#include <utility>
#include <chrono>
#include <limits.h>
#include <boost/thread.hpp>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <stack>
#include "utils.hpp"
using namespace std;
#define LARGE_INT 2147483647
/*Parameters*/

char param_graph_file_name[1024]="/home/chen/benchmarks/splex/2nd_dimacs/brock400_4.clq";
int param_s = 2;
int param_best = 999999;
int param_max_seconds = 120;
int param_cycle_iter = 4000;
unsigned int param_seed;
int thread_size = 10;

double gama1 = 0.8;
double gama2 = 2;
double alpha = 1.0;


/*original graph, the structure is kept the same,
 * but the id of vertices are renumbered, the original
 * id can be retrievaled by org_vid*/
int aaa;
int org_vnum;
int org_enum;
int org_fmt;
int* org_v_edge_cnt;
int** org_v_adj_vertex;
int *org_vid;	/*The real id of each original vertex*/

int** no_org_v_adj_vertex;		//no adjacent matrix data

int *red_orgid; /*the corresponding id in original graph*/
int *new_id;
int** red_v_adj_vertex;

int** k_decompos;
int *k_cnt_decompos;
int *org_decompos;
int *new_edge_count;
int **new_adj_tbl;
int **new_noadj_tbl;
int *layers;
int *v_in_layer;
int *global_layer_weight;

int min_vnum;
int cnt_layers;
int avg_mindeg;

int global_best_size = 0;
double global_best_time;
double gloabl_avg;
int* global_best_plex;



int low_group_size = 10;
int up_group_size;
int group_num;
int *local_in_group;
int *local_in_remain;
int *local_in_cand;
int *v_in_tem;
int *cur_g_deg;
int *deg_in_group;
int *avg_deg_group;
int *update_id_group;
int *group_weight;

vector<int> v_remain;
vector<vector<int>> v_group;

struct st
{
	int n,x;
	bool operator <(const st &a)const
	{
		if(a.x == x) return a.n < n; 
		return x < a.x;
	}
};

std::shared_mutex max_mutex;
std::shared_mutex group_mutex;
std::shared_mutex weight_mutex;
std::mutex printf_mutex;
std::atomic<bool> updated(false);

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
	}
	else if (global_best_size == value) {
		if (global_best_time > btime) {
			global_best_time = btime;
			for (int i = 0; i < global_best_size; i++) global_best_plex[i] = best_solution[i];
		}
	}
}

int is_update_global(int value) {
	std::shared_lock<std::shared_mutex> lock(max_mutex);
	if (global_best_size < value) return 1;
	else return 0;
}

int is_reduced(int deg) {
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
	if (v_group[i].size() == 0) return;
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
	//printf("before update size %ld\n", v_group.size());
	//for (int i = 0; i < v_group.size(); i++) printf("i %d size %ld weight %d\n", i, v_group[i].size(), group_weight[i]);
	for (int i = 0; i < v_group.size();) {
		for (int j = 0; j< v_group[i].size();) {
			v = v_group[i][j];
			if ((freqs[v] > 0) && (steps[v] > 0)) {
				if(((avg_step > steps[v]/freqs[v]) && (avg_deg_group[i] > deg_in_group[v])) || (v >= min_vnum)) {
					remove_from_group(i, j, v);
					//printf("remove v %d indeg %d avgdeg %d step/freqs %lld step %lld freqs %d avgstep %lld\n", v, deg_in_group[v], avg_deg_group[i], steps[v]/freqs[v], steps[v], freqs[v], avg_step);
				}
				else j++;
			}
			else j++;
		}
		//printf("i %d v_group %ld low_group_size %d \n", i, v_group[i].size(), low_group_size);
		if (v_group[i].size() < low_group_size) {
			//printf("erase i %d size %ld weight %d\n", i, v_group[i].size(), group_weight[i]);
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
	//printf("after update size %ld\n", v_group.size());
	//for (int i = 0; i < v_group.size(); i++) printf("i %d size %ld weight %d\n", i, v_group[i].size(), group_weight[i]);

	for (int i = 0; i < v_group.size(); i++) {
		if (v_group[i].size() > up_group_size) {
			if (v_group[i].size() % up_group_size == 0) group_size = v_group[i].size() / (v_group[i].size() / up_group_size);
			else group_size = v_group[i].size() / ((v_group[i].size() / up_group_size) + 1);
			//printf("i %d group size %ld up_group_size %d group_size %d %ld %d\n", i, v_group[i].size(), up_group_size, group_size, v_group[i].size() - up_group_size, (v_group[i].size() - up_group_size) > low_group_size);
		}
		while ((v_group[i].size() > up_group_size)  && ((v_group[i].size() - up_group_size) > low_group_size)) {
			vec_cand.clear();
			for (int j = 0; j < min_vnum; j++) v_in_tem[j] = -1;
			for (int j = 0; j < v_group[i].size(); j++) v_in_tem[v_group[i][j]] = j;
	        
			while(vec_cand.size() < group_size) {
				//int loc = rand() % v_group[i].size();
				//v = v_group[i][loc];
				//printf("v %d loc %d v_in %d group size %ld up_group_size %d\n", v, loc, v_in_tem[v], v_group[i].size(), up_group_size);
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
						//printf("adjv %d v_in %d group size %ld\n", adjv, v_in_tem[adjv], v_group[i].size());
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
				//printf("an group %ld temp %ld up_group_size %d\n", v_group[i].size(), vec_cand.size(), up_group_size);
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
			//printf("remain %ld cand %ld\n", v_group[i].size(), vec_cand.size());
		}
		update_degree_in_group(i, min_vnum);
	}
	//printf("bbb size %ld %ld\n", v_group.size(), vec_temp.size());
	//for (int x = 0; x < v_group.size(); x++) printf("i %d size %ld\n", x, v_group[x].size());
    
	
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
			// for (int j = 0; j < size; j++) {
			// 	if (max_indeg < cur_g_deg[j]) {
			// 		max_indeg = cur_g_deg[j];
			// 		id_group = j;
			// 		flag1 = 1;
			// 	}
			// }
			if(freqs[v] == 0) freqs[v] = freqs[v] + 1;
			for (int j = 0; j < size; j++) {
				if ((avg_step <= steps[v]/freqs[v]) && (avg_deg_group[j] <= cur_g_deg[j])) {
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
			//printf("add v %d indeg %d avgdeg %d step/freqs %lld step %lld freqs %d avgstep %lld\n", v, cur_g_deg[id_group], avg_deg_group[id_group], steps[v]/freqs[v], steps[v], freqs[v], avg_step);
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
	//printf("after update size %ld\n", v_group.size());
	//for (int i = 0; i < v_group.size(); i++) printf("i %d size %ld weight %d vnum %d\n", i, v_group[i].size(), group_weight[i], min_vnum);
	//printf("ccc\n");
	//for (int x = 0; x < v_group.size(); x++) printf("i %d size %ld\n", x, v_group[x].size());
	/*for (int i = 0; i < v_group.size(); i++) {
		printf("i %d size %ld avg_step %lld avgdeg %d\n", i, v_group[i].size(), avg_step, avg_deg_group[i]);
		for (int j = 0; j < v_group[i].size(); j++) {
			//if (freqs[v_group[i][j]] > 0) printf("%d(%d %lld) ", v_group[i][j], deg_in_group[v_group[i][j]], steps[v_group[i][j]]/freqs[v_group[i][j]]);
			//else printf("%d(%d %lld %d) ", v_group[i][j], deg_in_group[v_group[i][j]], steps[v_group[i][j]], freqs[v_group[i][j]]);
			printf("%d(%d %d) ", v_group[i][j], deg_in_group[v_group[i][j]], v_in_layer[v_group[i][j]]);
		}
		printf("\n");
	}*/
       /* printf("neig\n");
		for (int i = 0; i< min_vnum; i++) {
			printf("v %d(%d) :", i, local_in_group[i]);
			for (int j = 0; j < new_edge_count[i]; j++) {
				int neig = new_adj_tbl[i][j];
				if (neig >= min_vnum) break;
				printf("%d(%d) ", neig, local_in_group[neig]);
			}
			printf("\n");
		}*/
}


/**
 * load instances from  2nd Dimacs competetion
 *
 */
int load_clq_instance(char* filename){
	ifstream infile(filename);
	char line[1024];
	char tmps1[1024];
	char tmps2[1024];
	/*graph adjacent matrix*/
	int **gmat;
	if (!infile.is_open()){
		fprintf(stderr,"Can not find file %s\n", filename);
		return 0;
	}

	infile.getline(line,  1024);
	while (line[0] != 'p')	infile.getline(line,1024);
	sscanf(line, "%s %s %d %d", tmps1, tmps2, &org_vnum, &org_enum);

	gmat = new int*[org_vnum];
	for (int i = 0; i < org_vnum; i++){
		gmat[i] = new int[org_vnum];
		memset(gmat[i], 0, sizeof(int) * org_vnum);
	}

	int ecnt = 0;
	org_v_edge_cnt = new int[org_vnum];
	while (infile.getline(line, 1024)){
		int v1,v2;
		if (strlen(line) == 0)
			continue;
		if (line[0] != 'e')
			fprintf(stderr, "ERROR in line %d\n", ecnt+1);
		sscanf(line, "%s %d %d", tmps1, &v1, &v2);
		v1--,v2--;
		gmat[v1][v2] = 1;
		gmat[v2][v1] = 1;
		org_v_edge_cnt[v1]++;
		org_v_edge_cnt[v2]++;
		ecnt++;
	}
	assert(org_enum == ecnt);
	org_fmt = 0;
	org_vid = new int[org_vnum];
	org_v_adj_vertex = new int*[org_vnum];
	no_org_v_adj_vertex = new int*[org_vnum];	//no adjacent vertices matrix data
	for (int i = 0; i < org_vnum; i++){
		int adj_cnt = 0;
		int no_adj_cnt=0;
		org_v_adj_vertex[i] = new int[org_v_edge_cnt[i]];
		no_org_v_adj_vertex[i]= new int[org_vnum-org_v_edge_cnt[i]-1];  //no adjacent vertices data of i
		org_vid[i] = i+1;
		for (int j = 0; j < org_vnum; j++){
			if (gmat[i][j] == 1)
				org_v_adj_vertex[i][adj_cnt++] = j;
			else if (gmat[i][j] == 0 && (i != j)){
				no_org_v_adj_vertex[i][no_adj_cnt++] = j;		//no adjacent vertices data added to matrix
			}
		}
		assert(org_v_edge_cnt[i] == adj_cnt);
		delete[] gmat[i];
	}
	delete[] gmat;
	return 1;
}

/*load instances from  Stanford Large Network Dataset Collection
 * URL: http://snap.stanford.edu/data/*/
int load_snap_instance(char *filename){
	ifstream infile(filename);
	char line[1024];
	vector<pair<int,int> > *pvec_edges = new vector<pair<int, int> >();
	const int CONST_MAX_VE_NUM = 9999999;
	if (!infile.is_open()){
		fprintf(stderr,"Can not find file %s\n", filename);
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

/*load instances of metis format from
 * instances are download from
 * http://www.cc.gatech.edu/dimacs10/downloads.shtml
 */
int load_metis_instance(char* filename){
	ifstream infile(filename);
	string line;

	if (!infile.is_open()){
		fprintf(stderr,"Can not find file %s\n", filename);
		exit(0);
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
	fprintf(stderr, "splex -f <filename> -k <paramete s> -t <max seconds>  [-o optimum object] [-n threads]");
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
	/*check parameters*/
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

void load_instance(int argc, char **argv) {
	int load = 0;
	read_params(argc, argv);
	const char* fileext = file_suffix(param_graph_file_name);
	if (0 == strcmp(fileext, "graph")){
		load = load_metis_instance(param_graph_file_name);
	}else if(0 == strcmp(fileext, "txt")){
		load = load_snap_instance(param_graph_file_name);
	}else if(0 == strcmp(fileext, "clq")){
		load = load_clq_instance(param_graph_file_name);
	}
	if (load != 1){
		fprintf(stderr, "failed in loading graph %s\n",param_graph_file_name);
		exit(-1);
	}
}

class SplexClassic
{
public:

    int* time_stamp;
  	int total_add = 0;
	int sai_add = 0;
	int count_add = 0;
	int total_swap = 0;
	int sai_swap = 0;
	int count_swap = 0;
	int count_per = 0;
	int size_per = 0;
	int restart_pass = 0;
	int time_update = 0;

	/*reduced graph*/
	int red_vnum;
	int red_enum;
	int* red_v_edge_cnt;
	int red_min_deg ;
	int *freq;
	int *freq1;
	int *momentum;
	int *threshold;
	int *deposit;
	int cur_iter;
	int *cur_c_deg;
	int *cur_c_consat; /*cur_c_consat[v] Then number of saturated neighboors of v*/
	int cur_c_satu_num; //the number of saturate vertices
	int *is_in_c;
	int *plex_missing;		//plex_missing[v]: the number of non-adjacent vertices in cur_splex and v
	int *critical_missing;	//critical_missing[v]: the number of non-adjacent critical vertices in cur_splex and v
	int *unadj_satu_with_M2;
	int *M2_type;
	RandAccessList *vec_M1;
	RandAccessList *vec_M2;
	RandAccessList *cur_splex;
	RandAccessList *cur_cand;
	//clock_t start_time;
	int flag_remove = 1;
	//fast construct
	bool* can_add;
	bool* is_satu;
	int* remove_flag;
	/*final result*/
	int best_size;
	int* best_plex;
	//clock_t best_time;
	int best_found_iter;
	int total_iter;
	//clock_t total_time;
	int total_start_pass;
	int* local_opt_score;
	int local_best_size = 0;
	int cycle_iter;

	struct timespec start_time;
	struct timespec best_time;
	struct timespec total_time;
	double totime = 0;
	double the_best_time = 0;

	long long freq_inc;
	long long freq_old;
	long long *step_in_solution;
	long long *add_step;
	long long iter_step;

	int	start_group;
	int init_met;
	int *local_layer_weight;


	unsigned int rand_seed = 0,rand_seed_record = 0; 

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
		//assert(ral->vpos[vid] >= ral->vnum || ral->vlist[ral->vpos[vid]] != vid);
		ral->vlist[ral->vnum] = vid;
		ral->vpos[vid] = ral->vnum;
		ral->vnum++;
	}
	void ral_delete(RandAccessList *ral, int vid) {
		// assert(ral->vpos[vid] < ral->vnum);
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
	void ral_release(RandAccessList *ral) {
		delete[] ral->vlist;
		delete[] ral->vpos;
		delete ral;
	}

	/*
	int cmpfunc(const void * a, const void * b)
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
	*/

	inline bool ver_exist(RandAccessList *ral, int ver, int flagg) {
		bool flag = (ral->vpos[ver] < ral->vnum &&  ral->vlist[ral->vpos[ver]] == ver);
		//if (flag == false)
			//cout << param_graph_file_name << "ver exist " << param_seed << " " << flagg << " " << restart_pass << " " << cycle_iter << endl;
		return flag;
		//return 1;
	}

	inline bool ver_exist2(RandAccessList *ral, int ver) {
		bool flag = (ral->vpos[ver] < ral->vnum &&  ral->vlist[ral->vpos[ver]] == ver);
		return flag;
		//return 1;
	}

	/*recursively remove all the vertices with degree less or equal than reduce_deg,
 	* reconstructing the reduced graph  */
	// void reduce_graph(int reduce_deg){
	// 	// cout << "the reduce_deg is " << reduce_deg << "\t";
	// 	int critical_num = 0;
	
	// 	int rm_cnt = 0;
	// 	int *rmflag = new int[red_vnum];
	// 	int *remain_edge_count = new int[red_vnum];
	// 	/*Avoid unnecessary reduction*/
	// 	if (red_min_deg > reduce_deg){
	// 		return;
	// 	}
	// 	cout << "reduce" << param_graph_file_name << param_seed << endl;
	// 	memcpy(remain_edge_count, red_v_edge_cnt, sizeof(int) * red_vnum);
	// 	memset(rmflag, 0, sizeof(int) * red_vnum);
	// 	queue<int> rm_que;

	// 	for (int idx = 0; idx < red_vnum; idx++){
	// 		if (remain_edge_count[idx] <= reduce_deg){
	// 			if (remain_edge_count[idx] == reduce_deg) { critical_num++; }//count
	// 			rmflag[idx] = 1;
	// 			rm_que.push(idx);
	// 			rm_cnt++;
	// 		}
	// 	}
	// 	while(!rm_que.empty()){
	// 		int idx = rm_que.front();
	// 		rm_que.pop();
	// 		for (int i = 0; i < red_v_edge_cnt[idx]; i++){
	// 			int adjv = red_v_adj_vertex[idx][i];
	// 			remain_edge_count[adjv]--;
	// 			if (!rmflag[adjv] && remain_edge_count[adjv] <= reduce_deg){
	// 				if (remain_edge_count[adjv] == reduce_deg) { critical_num++; }//count
	// 				rmflag[adjv] = 1;
	// 				rm_que.push(adjv);
	// 				rm_cnt++;
	// 			}
	// 		}
	// 	}
	// 	/*rebuild the reduced graph*/
	// 	int rest_vnum = red_vnum - rm_cnt;
	// 	int *new_id = new int[red_vnum];
	// 	/*Mapp the vertex id to the original id(in the initial graph) */
	// 	int *org_id = new int[rest_vnum];
	// 	int count = 0; //count the rest vertices
	// 	int *last_freq = new int[rest_vnum];
	// 	int *last_momentum = new int[rest_vnum];
	// 	double *last_potential = new double[rest_vnum];
	// 	//resign new id to the rest of the vertices
	// 	for (int idx = 0; idx < red_vnum; idx++){
	// 		if (!rmflag[idx]){
	// 			new_id[idx] = count;
	// 			org_id[count] = red_orgid[idx];
	// 			last_freq[count] = freq[idx];
	// 			last_momentum[count] = momentum[idx];
	// 			count++;
	// 		}
	// 	}
	// 	/**/
	// 	int min_deg = LARGE_INT;
	// 	int n_edges = 0;
	// 	int **new_adj_tbl = new int*[rest_vnum];
	// 	int *new_edge_count = new int[rest_vnum];
	// 	for (int idx_prev = 0; idx_prev < red_vnum; idx_prev++){
	// 		if (rmflag[idx_prev]){
	// 			delete[] red_v_adj_vertex[idx_prev];
	// 			continue;
	// 		}
	// 		int idx_new = new_id[idx_prev];
	// 		new_adj_tbl[idx_new] = new int[remain_edge_count[idx_prev]];
	// 		new_edge_count[idx_new] = remain_edge_count[idx_prev];
	// 		int cnt = 0;
	// 		for (int i = 0; i < red_v_edge_cnt[idx_prev]; i++){
	// 			int vi_adj = red_v_adj_vertex[idx_prev][i];
	// 			if (!rmflag[vi_adj]){
	// 				new_adj_tbl[idx_new][cnt++] = new_id[vi_adj];
	// 				n_edges++;
	// 			}
	// 		}
	// 		if (new_edge_count[idx_new] < min_deg)
	// 			min_deg = new_edge_count[idx_new];
	// 		assert(cnt == remain_edge_count[idx_prev]);
	// 		delete[] red_v_adj_vertex[idx_prev];
	// 	}
	// 	assert(n_edges % 2 == 0);
	// 	/*assign to the new graph*/
	// 	delete[] red_v_adj_vertex;
	// 	red_v_adj_vertex = new_adj_tbl;
	// 	delete[] red_v_edge_cnt;
	// 	red_v_edge_cnt = new_edge_count;

	// 	red_vnum = rest_vnum;
	// 	red_enum = n_edges/2;

	// 	/*reset the minimum deg*/
	// 	red_min_deg = min_deg;

	// 	/*reset the original id map*/
	// 	delete[] red_orgid;
	// 	red_orgid = org_id;

	// 	/*reset the last join record*/
	// 	delete[] freq;
	// 	freq = last_freq;

	// 	delete[] momentum;
	// 	momentum = last_momentum;

	// 	delete[] new_id;	
	// 	delete[] rmflag;
	// 	delete[] remain_edge_count;

	// }

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
		//printf("org_vnum %d red_vnum %d\n", org_vnum, red_vnum);
		// if (is_update_group(red_vnum) && (!is_group_empty())) {
		// 	vector<int> vec_M1;
		// 	for (int i = 0; i < best_size; i++) {
		// 		int v = new_id[best_plex[i]];
		// 		vec_M1.push_back(v);
		// 	}
		// 	update_group(red_vnum, thread_id, vec_M1);
		// }
	}

	/*reinitial the data for a new start of the algorithm*/
	void init_search(int thread_id){
		red_min_deg = LARGE_INT;
		for (int v = 0; v < red_vnum; v++){
			if (red_v_edge_cnt[v] < red_min_deg)
				red_min_deg = red_v_edge_cnt[v];
		}
		/*init search data*/
		cur_iter = 0;
		iter_step = 0;
		cur_c_deg = new int[red_vnum];
		is_in_c = new int[red_vnum];
		memset(cur_c_deg, 0, sizeof(int) * red_vnum);
		memset(is_in_c, 0, sizeof(int) * red_vnum);
		freq = new int[red_vnum];
		memset(freq, 0, sizeof(int) * red_vnum);
		freq1 = new int[red_vnum];
		memset(freq1, 0, sizeof(int) * red_vnum);
		momentum = new int [red_vnum];
		memset(momentum, 0, sizeof(int) * red_vnum);
		local_opt_score = new int[red_vnum];
		memset(local_opt_score, 0, sizeof(int) * red_vnum);
		step_in_solution = new long long [red_vnum];
		memset(step_in_solution, 0, sizeof(long long) * red_vnum);
		add_step = new long long[red_vnum];
		memset(add_step, 0, sizeof(long long) * red_vnum);
		local_layer_weight = new int[cnt_layers];
		time_stamp = new int[org_vnum];
		cur_splex = ral_init(red_vnum);
		cur_cand = ral_init(red_vnum);
		vec_M1 = ral_init(red_vnum);
		vec_M2 = ral_init(red_vnum);
		M2_type = new int[red_vnum];
		memset(M2_type, 0, sizeof(int) * org_vnum);

		threshold = new int[red_vnum];
		deposit = new int[red_vnum];

		can_add = new bool[red_vnum];
		is_satu = new bool[red_vnum];

		plex_missing=new int[org_vnum]; 		//initialize plex_missing and critical_missing matrix 
		critical_missing=new int[org_vnum];
		remove_flag = new int[red_vnum];
		memset(remove_flag, 0, sizeof(int) * red_vnum);
		memset(plex_missing, 0, sizeof(int) * org_vnum);
		memset(critical_missing, 0, sizeof(int) * org_vnum);
		unadj_satu_with_M2=new int[org_vnum]; 
		memset(unadj_satu_with_M2, -1, sizeof(int) * org_vnum);
		/*init best found data*/
		best_size = 0;
		/*We could reduce this size*/
		best_plex = new int[red_vnum];
		//best_time = 0;
		best_found_iter = 0;
		total_start_pass = 0;

		//start_time = clock();
		//srand((unsigned int)param_seed);
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start_time);
		srand((unsigned int)thread_id);

		if (org_vnum == 400 || org_vnum == 4000) {
			flag_remove = 0;
		}
	}

	void restart_search(){
		local_best_size = 0;
		memset(time_stamp, 0, sizeof(int) * org_vnum);
		memset(plex_missing, 0, sizeof(int) * org_vnum);		//reset plex_missing and critical_missing when restart
		memset(critical_missing, 0, sizeof(int) * org_vnum);
		memset(unadj_satu_with_M2, -1, sizeof(int) * org_vnum);
		memset(M2_type, 0, sizeof(int) * org_vnum);
		memset(cur_c_deg, 0, sizeof(int) * red_vnum);
		memset(is_in_c, 0, sizeof(int) * red_vnum);
		memset(local_opt_score, 0, sizeof(int) * red_vnum);
		memset(add_step, 0, sizeof(long long) * red_vnum);
		ral_clear(cur_splex);
		ral_clear(cur_cand);	
		for (int i = 0; i < red_vnum; ++i){
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
		//printf("freq_inc %lld freq_thres %d red_vnum %d alpha %lf sumfreq %lld tierstep %lld\n", freq_inc, freq_thres, red_vnum, alpha, sumfreq, iter_step);
		if (freq_inc > freq_thres) {
			//printf("freq_inc %lld freq_old %lld red_vnum %d best_size %d\n", freq_inc, freq_old, red_vnum, best_size);
			for (int i = 0; i < best_size; i++) {
				int v = new_id[best_plex[i]];
				vec_M1.push_back(v);
				//printf("%d %lld %d %lld %d\n", v, iter_step - add_step[v], freq[v], iter_step, local_in_group[v]);
			}
			if(!is_group_empty()) {
				update_group_partition(red_vnum, vec_M1, freq, step_in_solution);
				time_update++;
			}
			freq_old = sumfreq;
		}
		
	}


	void add_without_change_deposit(int v){	//new add function without change deposit
		aaa = 1;
		//printf("add_without_change_deposit %d %d\n", v, is_in_c[v]);
		assert(!is_in_c[v]);
		is_in_c[v] = 1;
		add_step[v] = iter_step;

		ral_add(cur_splex, v);
		if (cur_c_deg[v] > 0){
			ral_delete(cur_cand, v);
		}
		// printf("before vec_M2\n");
		// for (int i = 0; i < vec_M2->vnum; i++) {
		// 	printf("%d ", vec_M2->vlist[i]);
		// }
		// printf("\n");

		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int adjv = red_v_adj_vertex[v][i];
			cur_c_deg[adjv]++;
			if (!is_in_c[adjv] && cur_c_deg[adjv] == 1){
				ral_add(cur_cand, adjv);
			}
		}
		ral_delete(vec_M1, v);
		for(int i = 0; i < red_vnum-red_v_edge_cnt[v]-1; i++){    //no-adjacent vertices times iteration 
			int no_adjv = no_org_v_adj_vertex[v][i];	//no-adjacent vertex[i] of v
			if(no_adjv >= red_vnum) printf("1 no_adjv %d red_vnum %d\n", no_adjv, red_vnum);
			plex_missing[no_adjv]++;	//plex_missing++
			if(!is_in_c[no_adjv]){			//no-adjacent vertices out of cur_splex,if their plex_missing up to bound, will change to another set
				if(plex_missing[no_adjv] == param_s+1) {  //k->k+1
					if(critical_missing[no_adjv] <= 1){	//M2 to M3
						if (ver_exist(vec_M2, no_adjv, 2)){
							ral_delete(vec_M2, no_adjv);
						}
					}
				}
				else if(plex_missing[no_adjv] == param_s) {    // k-1->k
					if(critical_missing[no_adjv] == 0) {	//M1 to M2
						if (ver_exist(vec_M1, no_adjv, 1)){
							ral_delete(vec_M1, no_adjv);
							ral_add(vec_M2, no_adjv);
							M2_type[no_adjv] = 1;
						}
					}
				}
			}
			else if(is_in_c[no_adjv]){	//if no-adjacent vertex belong to cur_splex and plex_missing change to bound,their no-adjacent vertices will change critical_missing
				if(plex_missing[no_adjv] == param_s-1) { //plex_missing up to k-1,become a critical node,critical_missing of no-adjacent vertices of it will plus 1
					for(int j=0; j < red_vnum-red_v_edge_cnt[no_adjv]-1; j++) {	//no-adjacent vertices of critical node
						int second_no_adjv = no_org_v_adj_vertex[no_adjv][j];
						if(second_no_adjv >= red_vnum) printf("2 second_no_adjv %d red_vnum %d\n", second_no_adjv, red_vnum);
						critical_missing[second_no_adjv]++;	//critical_missing++
						unadj_satu_with_M2[second_no_adjv]=no_adjv;
						if(critical_missing[second_no_adjv] == 2 && !is_in_c[second_no_adjv] && second_no_adjv != v){  // M2 to M3
							if(plex_missing[second_no_adjv] <= param_s){
								if (ver_exist(vec_M2, second_no_adjv, 2)) {
									ral_delete(vec_M2, second_no_adjv);
								}
							}
						}
						else if(critical_missing[second_no_adjv] == 1 && !is_in_c[second_no_adjv] && second_no_adjv != v){ //c_m(critical_missing)0->1
							if(plex_missing[second_no_adjv] == param_s){   //p_m(plex_missing)==k,but c_m(0->1),M2(second=1) to M2(second=0)
								if (ver_exist(vec_M2, second_no_adjv, 2) && M2_type[second_no_adjv] == 1) {
									M2_type[second_no_adjv] = 0;
								}
							}
							else if(plex_missing[second_no_adjv] <= param_s-1){    //M1 to M2
								if (ver_exist(vec_M1, second_no_adjv, 1)) {
									ral_delete(vec_M1, second_no_adjv);
									ral_add(vec_M2, second_no_adjv);
									M2_type[second_no_adjv] = 0;
								}
							}
						}
					}
				}
				else if(plex_missing[no_adjv] == param_s){	//k-1->k, the vertex will not be a critical node, the c_m of no-adjacent vertices of it will minus 1
					for(int j=0; j < red_vnum-red_v_edge_cnt[no_adjv]-1; j++) {	//no-adjacent vertices of this node
						int second_no_adjv = no_org_v_adj_vertex[no_adjv][j];
						if(second_no_adjv >= red_vnum) printf("3 second_no_adjv %d red_vnum %d\n", second_no_adjv, red_vnum);
						critical_missing[second_no_adjv]--;	//c_m--
						if(unadj_satu_with_M2[second_no_adjv]== no_adjv){
							unadj_satu_with_M2[second_no_adjv]=-1;
						}
						if(critical_missing[second_no_adjv] == 1 && !is_in_c[second_no_adjv] && second_no_adjv != v){ //c_m 2->1
							if(plex_missing[second_no_adjv] <= param_s){    //M3 to M2
								ral_add(vec_M2, second_no_adjv);
								M2_type[second_no_adjv] = 0;
								if (unadj_satu_with_M2[second_no_adjv] == -1) {
									for (int k = 0; k < red_vnum - red_v_edge_cnt[second_no_adjv] - 1; k++) {
										if (is_in_c[no_org_v_adj_vertex[second_no_adjv][k]] && plex_missing[no_org_v_adj_vertex[second_no_adjv][k]] == param_s - 1) {
											unadj_satu_with_M2[second_no_adjv] = no_org_v_adj_vertex[second_no_adjv][k];
											break;
										}
									}
								}
							}
						}
						else if(critical_missing[second_no_adjv] == 0 && !is_in_c[second_no_adjv] && second_no_adjv != v){  //c_m 1->0
							if(plex_missing[second_no_adjv] <= param_s-1){	// M2 to M1
								if (ver_exist(vec_M2, second_no_adjv, 2)) {
									ral_delete(vec_M2, second_no_adjv);
									ral_add(vec_M1, second_no_adjv);
								}
								
							}
							else if(plex_missing[second_no_adjv] == param_s){ // M2 to M2
								if (ver_exist(vec_M2, second_no_adjv, 2) && M2_type[second_no_adjv] == 0) {
									M2_type[second_no_adjv] = 1;
								}
							}
						}
					}
				}

			}
		}
		if(plex_missing[v] == param_s-1){	//v is a critical node and now add to cur_splex
			for(int j=0; j < red_vnum-red_v_edge_cnt[v]-1; j++) {	//no-adjacent vertices of it
				int no_adjv = no_org_v_adj_vertex[v][j];
				if(no_adjv >= red_vnum) printf("4 no_adjv %d red_vnum %d\n", no_adjv, red_vnum);
				critical_missing[no_adjv]++;	//c_m++
				unadj_satu_with_M2[no_adjv]=v;
				if(critical_missing[no_adjv] == 2 && !is_in_c[no_adjv] && no_adjv != v){ //c_m 1->2  
					if(plex_missing[no_adjv] <= param_s){// M2 to M3
						if (ver_exist(vec_M2, no_adjv, 2)) {
							ral_delete(vec_M2, no_adjv);
						}
					}
				}
				else if(critical_missing[no_adjv] == 1 && !is_in_c[no_adjv] && no_adjv != v){ //c_m 0->1
					if(plex_missing[no_adjv] == param_s){   //M2 to M2
						if (ver_exist(vec_M2, no_adjv, 2) && M2_type[no_adjv] == 1) {
							M2_type[no_adjv] = 0;
						}
					}
					else if(plex_missing[no_adjv] <= param_s-1){    //M1 to M2
						if (ver_exist(vec_M1, no_adjv, 1)) {
							ral_delete(vec_M1, no_adjv);
							ral_add(vec_M2, no_adjv);
							M2_type[no_adjv] = 0;
						}
					}
				}
			}
		}
		
		
		// printf("after vec_M2\n");
		// for (int i = 0; i < vec_M2->vnum; i++) {
		// 	printf("%d ", vec_M2->vlist[i]);
		// }
		// printf("\n");
	}

	int bms_thre_pro(int v) {
		int count = 0;
		for (int i = 0; i < 50; i++) {
			if (freq[v] > freq[rand_r(&rand_seed) % red_vnum]) {
				count++;
			}	
		}
		return count;//��count�ܴ�˵��freq��С����threshold��++
	}

	void add_cur_vertex(int v){		//old add function used in init solution create
		aaa = 2;
		//printf("add_cur_vertex %d %d\n", v, is_in_c[v]);
		assert(!is_in_c[v]);

		is_in_c[v] = 1;
		deposit[v] = 0;
		threshold[v] = (threshold[v] + 1) % 3;
		add_step[v] = iter_step;
		ral_add(cur_splex, v);
		if (cur_c_deg[v] > 0){
			ral_delete(cur_cand, v);
		}

	

		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int adjv = red_v_adj_vertex[v][i];
			++deposit[adjv];
			cur_c_deg[adjv]++;
			if (!is_in_c[adjv] && cur_c_deg[adjv] == 1){
				ral_add(cur_cand, adjv);
			}
		}
	}

	void add_cur_vertex_new(int v){   //new add function used in M1 and M3
		aaa = 3;
		//printf("add_cur_vertex_new %d %d\n", v, is_in_c[v]);
		assert(!is_in_c[v]);

		is_in_c[v] = 1;
		deposit[v] = 0;
		threshold[v] = threshold[v] + 1;
		add_step[v] = iter_step;
		if (threshold[v] >= 3) {
			int countttt = bms_thre_pro(v);
			if (countttt > 40) {
				threshold[v] = 1;
			}
			else {
				threshold[v] = 0;
			}
		}
		ral_add(cur_splex, v);
		if (cur_c_deg[v] > 0){
			ral_delete(cur_cand, v);
		}

		// printf("before vec_M2\n");
		// for (int i = 0; i < vec_M2->vnum; i++) {
		// 	printf("%d ", vec_M2->vlist[i]);
		// }
		// printf("\n");

		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int adjv = red_v_adj_vertex[v][i];
			++deposit[adjv];
			cur_c_deg[adjv]++;
			if (!is_in_c[adjv] && cur_c_deg[adjv] == 1){
				ral_add(cur_cand, adjv);
			}
		}
		for(int i = 0; i < red_vnum-red_v_edge_cnt[v]-1; i++){   //no-adjacent vertices times iteration  
			int no_adjv = no_org_v_adj_vertex[v][i];	//no-adjacent vertex[i] of v
			if(no_adjv >= red_vnum) printf("5 no_adjv %d red_vnum %d\n", no_adjv, red_vnum);
			plex_missing[no_adjv]++;	//plex_missing++
			if(!is_in_c[no_adjv]){	//no-adjacent vertices out of cur_splex,if their plex_missing up to bound, will change to another set
				if(plex_missing[no_adjv] == param_s+1) {  //k->k+1
					if(critical_missing[no_adjv] <= 1){	//M2 to M3
						if (ver_exist(vec_M2, no_adjv, 2)) {
							ral_delete(vec_M2, no_adjv);
						}
					}
				}
				else if(plex_missing[no_adjv] == param_s) {    // k-1->k
					if(critical_missing[no_adjv] == 0) {	//M1 to M2
						if (ver_exist(vec_M1, no_adjv, 1)) {
							ral_delete(vec_M1, no_adjv);
							ral_add(vec_M2, no_adjv);
							M2_type[no_adjv] = 1;
						}
					}
				}
			}
			else if(is_in_c[no_adjv]){	//if no-adjacent vertex belong to cur_splex and plex_missing change to bound,their no-adjacent vertices will change critical_missing
				if(plex_missing[no_adjv] == param_s-1) {	//plex_missing up to k-1,become a critical node,critical_missing of no-adjacent vertices of it will plus 1
					for(int j=0; j < red_vnum-red_v_edge_cnt[no_adjv]-1; j++) {	//no-adjacent vertices of critical node
						int second_no_adj = no_org_v_adj_vertex[no_adjv][j];
						if(second_no_adj >= red_vnum) printf("6 second_no_adj %d red_vnum %d\n", second_no_adj, red_vnum);
						critical_missing[second_no_adj]++;	//critical_missing++
						unadj_satu_with_M2[second_no_adj]=no_adjv;
						if(critical_missing[second_no_adj] == 2 && !is_in_c[second_no_adj] && second_no_adj != v){  // M2 to M3
							if(plex_missing[second_no_adj] <= param_s){
								if (ver_exist(vec_M2, second_no_adj, 2)) {
									ral_delete(vec_M2, second_no_adj);
								}
							}
						}
						else if(critical_missing[second_no_adj] == 1 && !is_in_c[second_no_adj] && second_no_adj != v){	//c_m(critical_missing)0->1
							if(plex_missing[second_no_adj] == param_s){   //p_m(plex_missing)==k,but c_m(0->1),M2(second=1) to M2(second=0)
								if (ver_exist(vec_M2, second_no_adj, 2) && M2_type[second_no_adj] == 1) {
									M2_type[second_no_adj] = 0;
								}
							}
							else if(plex_missing[second_no_adj] <= param_s-1){    //M1 to M2
								if (ver_exist(vec_M1, second_no_adj, 1)) {
									ral_delete(vec_M1, second_no_adj);
									ral_add(vec_M2, second_no_adj);
									M2_type[second_no_adj] = 0;
								}
							}
						}
					}
				}
				else if(plex_missing[no_adjv] == param_s){	///�Ƿ�Ϊ����Ҫɾ���ĵ㣿�����Ƴ������Ƿ��� plex_missing == param_s	//k-1->k, the vertex will not be a critical node, the c_m of no-adjacent vertices of it will minus 1
					for(int j=0; j < red_vnum-red_v_edge_cnt[no_adjv]-1; j++) {		//no-adjacent vertices of this node
						int second_no_adj = no_org_v_adj_vertex[no_adjv][j];
						if(second_no_adj >= red_vnum) printf("7 second_no_adj %d red_vnum %d\n", second_no_adj, red_vnum);
						critical_missing[second_no_adj]--;		//c_m-
						if(unadj_satu_with_M2[second_no_adj]==no_adjv){
							unadj_satu_with_M2[second_no_adj]=-1;
						}
						if(critical_missing[second_no_adj] == 1 && !is_in_c[second_no_adj] && second_no_adj != v){		//c_m 2->1
							if(plex_missing[second_no_adj] <= param_s){    //M3 to M2
								ral_add(vec_M2, second_no_adj);
								M2_type[second_no_adj] = 0;
								int curid = second_no_adj;
								if (unadj_satu_with_M2[curid] == -1) {
									for (int k = 0; k < red_vnum - red_v_edge_cnt[curid] - 1; k++) {
										if (is_in_c[no_org_v_adj_vertex[curid][k]] && plex_missing[no_org_v_adj_vertex[curid][k]] == param_s - 1) {
											unadj_satu_with_M2[curid] = no_org_v_adj_vertex[curid][k];
											break;
										}
									}
								}
							}
						}
						else if(critical_missing[second_no_adj] == 0 && !is_in_c[second_no_adj] && second_no_adj != v){  // M2 to M1
							if(plex_missing[second_no_adj] == param_s){ // M2 to M2
								if (ver_exist(vec_M2, second_no_adj, 2) && M2_type[second_no_adj] == 0) {
									M2_type[second_no_adj] = 1;
								}
								
							}
							else if(plex_missing[second_no_adj] <= param_s-1){
								if (ver_exist(vec_M2, second_no_adj, 2)) {
									ral_delete(vec_M2, second_no_adj);
									ral_add(vec_M1, second_no_adj);
								}
							}
						}
					}
				}
			}
		}
		if(plex_missing[v] == param_s-1){ 	//v is a critical node and now add to cur_splex
			for(int j=0; j < red_vnum-red_v_edge_cnt[v]-1; j++) {	//no-adjacent vertices of it
				int second_no_adj = no_org_v_adj_vertex[v][j];
				if(second_no_adj >= red_vnum) printf("8 second_no_adj %d red_vnum %d\n", second_no_adj, red_vnum);
				critical_missing[second_no_adj]++;	//c_m++
				unadj_satu_with_M2[second_no_adj]=v;
				if(critical_missing[second_no_adj] == 2 && !is_in_c[second_no_adj] && second_no_adj != v){  // M2 to M3
					if(plex_missing[second_no_adj] <= param_s){
						if (ver_exist(vec_M2, second_no_adj, 2)) {
							ral_delete(vec_M2, second_no_adj);
						}
					}
				}
				else if(critical_missing[second_no_adj] == 1 && !is_in_c[second_no_adj] && second_no_adj != v){
					if(plex_missing[second_no_adj] == param_s){   //M2 to M2
						if (ver_exist(vec_M2, second_no_adj, 2) && M2_type[second_no_adj] == 1) {
							M2_type[second_no_adj] = 0;
						}
					}
					else if(plex_missing[second_no_adj] <= param_s-1){    //M1 to M2
						if (ver_exist(vec_M1, second_no_adj, 1)) {
							ral_delete(vec_M1, second_no_adj);
							ral_add(vec_M2, second_no_adj);
							M2_type[second_no_adj] = 0;
						}
					}
				}
			}
		}
	
		// printf("after vec_M2\n");
		// for (int i = 0; i < vec_M2->vnum; i++) {
		// 	printf("%d ", vec_M2->vlist[i]);
		// }
		// printf("\n");
	}

	void remove_cur_vertex_new(int v){   //new remove function
		aaa = 4;
		//printf("remove_cur_vertex_new %d %d\n", v, is_in_c[v]);
		assert(is_in_c[v]);

		int tmpflag=0;
		if(plex_missing[v] == param_s-1) tmpflag=1;		//judge the vertex will be removed whether is a critical,if true, tmpflag=1

		is_in_c[v] = 0;
		deposit[v] = 0;
		step_in_solution[v] += iter_step - add_step[v];
		ral_delete(cur_splex, v);
		if (cur_c_deg[v] > 0){
			ral_add(cur_cand, v);
		}

		// printf("before vec_M2\n");
		// for (int i = 0; i < vec_M2->vnum; i++) {
		// 	printf("%d ", vec_M2->vlist[i]);
		// }
		// printf("\n");

		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int adjv = red_v_adj_vertex[v][i];
			cur_c_deg[adjv]--;
			if (!is_in_c[adjv] && cur_c_deg[adjv] == 0){
				ral_delete(cur_cand, adjv);
			}
		}
		for(int i = 0; i < red_vnum-red_v_edge_cnt[v]-1; i++){     	//no-adjacent vertices times iteration 
			int no_adjv = no_org_v_adj_vertex[v][i];	//no-adjacent vertex[i] of v
			if(no_adjv >= red_vnum) printf("9 no_adjv %d red_vnum %d\n", no_adjv, red_vnum);
			plex_missing[no_adjv]--;		//plex_missing--
			if(!is_in_c[no_adjv]){	//no-adjacent vertices out of cur_splex,if their plex_missing down to bound, will change to another set
				if(plex_missing[no_adjv] == param_s) {    //M3 to M2
					if(critical_missing[no_adjv] == 1){
						ral_add(vec_M2, no_adjv);
						M2_type[no_adjv] = 0;
						if (unadj_satu_with_M2[no_adjv] == -1) {
							for (int k = 0; k < red_vnum - red_v_edge_cnt[no_adjv] - 1; k++) {
								if (is_in_c[no_org_v_adj_vertex[no_adjv][k]] && plex_missing[no_org_v_adj_vertex[no_adjv][k]] == param_s - 1) {
									unadj_satu_with_M2[no_adjv] = no_org_v_adj_vertex[no_adjv][k];
									break;
								}
							}
						}
					}
					else if(critical_missing[no_adjv] == 0) {
						ral_add(vec_M2, no_adjv);
						M2_type[no_adjv] = 1;
					}
				}
				else if(plex_missing[no_adjv] == param_s-1) {  //M2 to M1
					if(critical_missing[no_adjv] == 0){
						if (ver_exist(vec_M2, no_adjv, 2)) {
							ral_delete(vec_M2, no_adjv);
							ral_add(vec_M1, no_adjv);
						}
					}
				}
			}

			else if(is_in_c[no_adjv]){	//if no-adjacent vertex belong to cur_splex and plex_missing change to bound(k->k-1),their no-adjacent vertices will change critical_missing
				if(plex_missing[no_adjv] == param_s-2) {	//plex_missing down to k-2,become a not critical node,critical_missing of no-adjacent vertices of it will minus 1
					for(int j=0; j < red_vnum-red_v_edge_cnt[no_adjv]-1; j++) {		//no-adjacent vertices of critical node
						int second_no_adjv = no_org_v_adj_vertex[no_adjv][j];
						if(second_no_adjv >= red_vnum) printf("10 second_no_adjv %d red_vnum %d\n", second_no_adjv, red_vnum);
						critical_missing[second_no_adjv]--;		//critical_missing--
						if(unadj_satu_with_M2[second_no_adjv]==no_adjv){
							unadj_satu_with_M2[second_no_adjv]=-1;
						}
						if(critical_missing[second_no_adjv] == 1 && !is_in_c[second_no_adjv] && second_no_adjv != v){
							if(plex_missing[second_no_adjv] <= param_s){    //M3 to M2
								ral_add(vec_M2, second_no_adjv);
								M2_type[second_no_adjv] = 0;
								if (unadj_satu_with_M2[second_no_adjv] == -1) {
									for (int k = 0; k < red_vnum - red_v_edge_cnt[second_no_adjv] - 1; k++) {
										if (is_in_c[no_org_v_adj_vertex[second_no_adjv][k]] && plex_missing[no_org_v_adj_vertex[second_no_adjv][k]] == param_s - 1) {
											unadj_satu_with_M2[second_no_adjv] = no_org_v_adj_vertex[second_no_adjv][k];
											break;
										}
									}
								}
							}
						}
						else if(critical_missing[second_no_adjv] == 0 && !is_in_c[second_no_adjv] && second_no_adjv != v){  // M2 to M1
							if(plex_missing[second_no_adjv] == param_s){ // M2 to M2
								if (ver_exist(vec_M2, second_no_adjv, 2) && M2_type[second_no_adjv] == 0) {
									M2_type[second_no_adjv] = 1;
								}
							}
							else if(plex_missing[second_no_adjv] <= param_s-1){
								if (ver_exist(vec_M2, second_no_adjv, 2)) {
									ral_delete(vec_M2, second_no_adjv);
									ral_add(vec_M1, second_no_adjv);
								}
							}
						}
					}
				}
				else if(plex_missing[no_adjv] == param_s-1) {	//plex_missing down to k-1(k->k-1),become a critical node,critical_missing of no-adjacent vertices of it will plus 1
					for(int j=0; j < red_vnum-red_v_edge_cnt[no_adjv]-1; j++) {	//no-adjacent vertices of critical node
						int second_no_adjv = no_org_v_adj_vertex[no_adjv][j];
						if(second_no_adjv >= red_vnum) printf("11 second_no_adjv %d red_vnum %d\n", second_no_adjv, red_vnum);
						critical_missing[second_no_adjv]++;	//c_m++
						unadj_satu_with_M2[second_no_adjv]=no_adjv;
						if(critical_missing[second_no_adjv] == 2 && !is_in_c[second_no_adjv] && second_no_adjv != v){  // M2 to M3
							if(plex_missing[second_no_adjv] <= param_s){
								if (ver_exist(vec_M2, second_no_adjv, 2)) {
									ral_delete(vec_M2, second_no_adjv);
								}
							}
						}
						else if(critical_missing[second_no_adjv] == 1 && !is_in_c[second_no_adjv] && second_no_adjv != v){
							if(plex_missing[second_no_adjv] == param_s){   //M2 to M2
								if (ver_exist(vec_M2, second_no_adjv, 2) && M2_type[second_no_adjv] == 1) {
									M2_type[second_no_adjv] = 0;
								}
								
							}
							else if(plex_missing[second_no_adjv] <= param_s-1){    //M1 to M2
								if (ver_exist(vec_M1, second_no_adjv, 1)) {
									ral_delete(vec_M1, second_no_adjv);
									ral_add(vec_M2, second_no_adjv);
									M2_type[second_no_adjv] = 0;
								}
							}
						}
					}
				}
			}
		}
		if(tmpflag == 1) {	//if the node removed is a critical node before,c_m of no-adjacent vertices of it will minus 1
			for(int j=0; j < red_vnum-red_v_edge_cnt[v]-1; j++) {	//no-adjacent vertices of critical node
				int second_no_adjv = no_org_v_adj_vertex[v][j];
				if(second_no_adjv >= red_vnum) printf("12 second_no_adjv %d red_vnum %d\n", second_no_adjv, red_vnum);
				critical_missing[second_no_adjv]--;	//c_m--
				if(unadj_satu_with_M2[second_no_adjv] == v){
					unadj_satu_with_M2[second_no_adjv]=-1;
				}
				if(critical_missing[second_no_adjv] == 1 && !is_in_c[second_no_adjv] && second_no_adjv != v){
					if(plex_missing[second_no_adjv] <= param_s){    //M3 to M2
						ral_add(vec_M2, second_no_adjv);
						M2_type[second_no_adjv] = 0;
						if (unadj_satu_with_M2[second_no_adjv] == -1) {
							for (int k = 0; k < red_vnum - red_v_edge_cnt[second_no_adjv] - 1; k++) {
								if (is_in_c[no_org_v_adj_vertex[second_no_adjv][k]] && plex_missing[no_org_v_adj_vertex[second_no_adjv][k]] == param_s - 1) {
									unadj_satu_with_M2[second_no_adjv] = no_org_v_adj_vertex[second_no_adjv][k];
									break;
								}
							}
						}
					}
				}
				else if(critical_missing[second_no_adjv] == 0 && !is_in_c[second_no_adjv] && second_no_adjv != v){  // M2 to M1
					if(plex_missing[second_no_adjv] <= param_s-1){
						if (ver_exist(vec_M2, second_no_adjv, 2)) {
							ral_delete(vec_M2, second_no_adjv);
							ral_add(vec_M1, second_no_adjv);
						}
						
					}
					else if(plex_missing[second_no_adjv] == param_s){ // M2 to M2
						if (ver_exist(vec_M2, second_no_adjv, 2) && M2_type[second_no_adjv] == 0) {
							M2_type[second_no_adjv] = 1;
						}
						
					}
				}
			}
		}	
		if(critical_missing[v]==0 && plex_missing[v]<=param_s-1) 	//add the removed vertex to the M(1/2/3) set
			ral_add(vec_M1, v);
		else if(critical_missing[v]==0 && plex_missing[v]==param_s) {
			ral_add(vec_M2, v);
			M2_type[v] = 1;
		}
		else if(critical_missing[v] == 1 && plex_missing[v] <= param_s) {
			ral_add(vec_M2, v);
			M2_type[v] = 0;
			if(unadj_satu_with_M2[v] == -1){
				for(int k = 0; k < red_vnum-red_v_edge_cnt[v]-1; k++){
					if (no_org_v_adj_vertex[v][k] >= red_vnum) printf("13 no_org_v_adj_vertex %d red_vnum %d\n", no_org_v_adj_vertex[v][k], red_vnum);
					if(is_in_c[no_org_v_adj_vertex[v][k]] && plex_missing[no_org_v_adj_vertex[v][k]] == param_s-1){
						unadj_satu_with_M2[v]=no_org_v_adj_vertex[v][k];
						break;
					}
				}
			}

		}
	
		// printf("after vec_M2\n");
		// for (int i = 0; i < vec_M2->vnum; i++) {
		// 	printf("%d ", vec_M2->vlist[i]);
		// }
		// printf("\n");
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
		memset(local_layer_weight, 0, sizeof(int) * cnt_layers);
		for (int i = 0; i < cur_splex->vnum; i++){
			int v = cur_splex->vlist[i];
			best_plex[i] = red_orgid[v];
			local_layer_weight[v_in_layer[v]]++;
		}
		//update_layer_weight(local_layer_weight);
		//if (init_met == 1) update_group_weight(start_group);
		//best_time = clock();
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &best_time);
		the_best_time = (double)((best_time.tv_sec - start_time.tv_sec) * 1000000 + (double)(best_time.tv_nsec - start_time.tv_nsec) / 1000) / 1000000;
		best_found_iter = cur_iter;
		if (is_update_global(best_size)) update_global(best_size, the_best_time, best_plex);
	}

	void record_local_best() {
		for (int i = 0; i < red_vnum; i++) {
			local_opt_score[i] = cur_c_deg[i];
		}
	}

	void search_frequency_init() {
		vector<int> vec_M1;
		bool* adj_flag = new bool[red_vnum];
		bool* satu_adj_flag = new bool[red_vnum];
		memset(can_add, 0, sizeof(bool) * red_vnum);
		memset(is_satu, 0, sizeof(bool) * red_vnum);
		int leastfreq = LARGE_INT;
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
		assert (!vec_M1.empty());
		int vstart = vec_M1[rand_r(&rand_seed)%vec_M1.size()];
		add_cur_vertex(vstart);   //use old add function here
		for (int i = 0; i < red_v_edge_cnt[vstart]; ++i){
			can_add[red_v_adj_vertex[vstart][i]] = true;
		}
		while(1){
			vec_M1.clear();
			leastfreq = LARGE_INT;
			for (int i = 0; i < red_vnum; ++i){
				if (can_add[i]){
					if (freq[i] < leastfreq){
						vec_M1.clear();
						vec_M1.push_back(i);
						leastfreq = freq[i];
					}else if(freq[i] == leastfreq){
						vec_M1.push_back(i);
					}
				}
			}
			if (vec_M1.empty()){
				break;
			}
			assert (!vec_M1.empty());
			iter_step++;
			int vadd = vec_M1[rand_r(&rand_seed)%vec_M1.size()];
			add_cur_vertex(vadd);
			can_add[vadd] = false;
			memset(adj_flag, 0, sizeof(bool) * red_vnum);
			for (int i = 0; i < red_v_edge_cnt[vadd]; ++i){
				adj_flag[red_v_adj_vertex[vadd][i]] = true;
			}
			for (int i = 0; i < red_vnum; ++i){
				if (!adj_flag[i]){
					if (is_in_c[i] && !is_satu[i] && Is_Saturated(i)){
						// i is just saturated
						is_satu[i] = true;
						memset(satu_adj_flag, 0, sizeof(bool) * red_vnum);
						for (int j = 0; j < red_v_edge_cnt[i]; ++j){
							satu_adj_flag[red_v_adj_vertex[i][j]] = true;
						}
						for (int j = 0; j < red_vnum; ++j){
							if (can_add[j] && !satu_adj_flag[j]){
								can_add[j] = false;
							}
						}
					}else if (!is_in_c[i] && can_add[i] && cur_c_deg[i] <= cur_splex->vnum - param_s){
						can_add[i] = false;
					}
				}
			}

		}
		init_met = 2;
	}

	void search_group_init(){
		int satcon, sat_size, vadd;
        vector<int> vec_M1;
		bool* adj_flag = new bool[red_vnum];
		bool* satu_adj_flag = new bool[red_vnum];
		memset(can_add, 0, sizeof(bool) * red_vnum);
		memset(is_satu, 0, sizeof(bool) * red_vnum);
		int leastfreq = LARGE_INT;

		pair<int, int> start = get_start_vertex();
		start_group = start.first;
		//printf("search_group_init start v %d red_vnum %d\n", start.second, red_vnum);
		add_cur_vertex(start.second);
		for (int i = 0; i < red_v_edge_cnt[start.second]; ++i){
			can_add[red_v_adj_vertex[start.second][i]] = true;
		}
		while(1){
			vec_M1.clear();
			leastfreq = LARGE_INT;
			for (int i = 0; i < red_vnum; ++i){
				if (!is_v_in_group(i)) continue;
				if (can_add[i]){
					if (freq[i] < leastfreq){
						vec_M1.clear();
						vec_M1.push_back(i);
						leastfreq = freq[i];
					}else if(freq[i] == leastfreq){
						vec_M1.push_back(i);
					}
				}
			}
			if (vec_M1.empty()){
				break;
			}
			assert (!vec_M1.empty());
			iter_step++;
			int vadd = vec_M1[rand_r(&rand_seed)%vec_M1.size()];
			//printf("add %d pos %d size %d vlist %d\n", vadd, cur_splex->vpos[vadd], cur_splex->vnum, cur_splex->vlist[cur_splex->vpos[vadd]]);
			//for (int x = 0; x < cur_splex->vnum; x++) printf("%d(%d %d) ", cur_splex->vlist[x], x, cur_splex->vpos[cur_splex->vlist[x]]);
			//printf("\n");
			add_cur_vertex(vadd);
			can_add[vadd] = false;
			memset(adj_flag, 0, sizeof(bool) * red_vnum);
			for (int i = 0; i < red_v_edge_cnt[vadd]; ++i){
				adj_flag[red_v_adj_vertex[vadd][i]] = true;
			}
			for (int i = 0; i < red_vnum; ++i){
				if (!adj_flag[i]){
					if (is_in_c[i] && !is_satu[i] && Is_Saturated(i)){
						// i is just saturated
						is_satu[i] = true;
						memset(satu_adj_flag, 0, sizeof(bool) * red_vnum);
						for (int j = 0; j < red_v_edge_cnt[i]; ++j){
							satu_adj_flag[red_v_adj_vertex[i][j]] = true;
						}
						for (int j = 0; j < red_vnum; ++j){
							if (can_add[j] && !satu_adj_flag[j]){
								can_add[j] = false;
							}
						}
					}else if (!is_in_c[i] && can_add[i] && cur_c_deg[i] <= cur_splex->vnum - param_s){
						can_add[i] = false;
					}
				}
			}

		}
		// for (int i = 0; i < cur_splex->vnum; i++) 
		// {
		// 	if (is_v_in_group(cur_splex->vlist[i]) == 0) printf("aaaa\n");
		// }
		init_met = 1;
	}


	void fast_init_solution(){
		// if((!is_group_empty()) && (rand_r(&rand_seed)%100 < 60)) {
		// 	search_group_init();
		// }
		// else {
		// 	search_frequency_init(); 
		// }
		search_frequency_init();
		if (cur_splex->vnum == red_vnum) goto end;
		while( cur_splex->vnum < param_s){
			if (cur_splex->vnum == red_vnum) break;
			int vrand = rand_r(&rand_seed) % red_vnum;
			//printf("fast init while vrand %d\n", vrand);
			while (is_in_c[vrand]) vrand = rand_r(&rand_seed) % red_vnum;
			//printf("fast init after while vrand %d\n", vrand);
			add_cur_vertex(vrand);
		}
	end:
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

	/*find the only unadjacent saturated vertex of v*/
	int find_unadj_satu(int v){
		//printf("find_unadj_satu %d %d\n", v, is_in_c[v]);
		assert(!is_in_c[v]);
		int *mark = new int[cur_splex->vnum];
		int v_rt = -1;
		memset(mark, 0, sizeof(int) * cur_splex->vnum);
		for (int i = 0; i < red_v_edge_cnt[v]; i++){//�ھ��ڽ���
			int vcur = red_v_adj_vertex[v][i];
			if (is_in_c[vcur]){
				mark[cur_splex->vpos[vcur]] = 1;
			}
		}
		for (int i = 0; i < cur_splex->vnum; i++){
			int vin = cur_splex->vlist[i];
			if (Is_Saturated(vin) && mark[i] == 0){
				v_rt = vin;
				break;
			}
		}
		delete[] mark;
		return v_rt;
	}

	/*get a random vertex of C\N_C(v)*/
	int random_unadj_with_exception(int v, int vexception){
		//printf("random_unadj_with_exception %d %d\n", v, is_in_c[v]);
		assert(!is_in_c[v]);
		int *mark = new int[cur_splex->vnum];
		vector<int> vec_unadj;
		memset(mark, 0, sizeof(int) * cur_splex->vnum);
		for (int i = 0; i < red_v_edge_cnt[v]; i++){
			int vcur = red_v_adj_vertex[v][i];
			if (is_in_c[vcur]){
				mark[cur_splex->vpos[vcur]] = 1;
			}
		}
		for (int i = 0; i < cur_splex->vnum; i++){
			if (!mark[i] && cur_splex->vlist[i] != vexception)
				vec_unadj.push_back(cur_splex->vlist[i]);
		}
		delete[] mark;
		if (vec_unadj.size() == 0)
			return -1;
		else
			return vec_unadj[rand_r(&rand_seed) % vec_unadj.size()];
	}

	int get_most_momentum(vector<int>& perturb_set){
		int min_momentum_2 = -999999999;
		vector<int> cand;
		vector<int> cand2;
		cand.push_back(perturb_set[0]);
		int min_momentum = momentum[perturb_set[0]];
		for (int i = 1, size = perturb_set.size(); i < size; ++i){
			int score = momentum[perturb_set[i]];
			if (score < min_momentum && min_momentum_2 == -999999999) {
				min_momentum_2 = score;
				cand2.push_back(perturb_set[i]);
			}
			else if (score > min_momentum) {
				min_momentum_2 = min_momentum;
				cand2.clear();
				for (int i = 0; i < cand.size(); i++) {
					cand2.push_back(cand[i]);
				}

				min_momentum = score;
				cand.clear();
				cand.push_back(perturb_set[i]);
			}
			else if (score < min_momentum && score > min_momentum_2) {
				min_momentum_2 = score;
				cand2.clear();
				cand2.push_back(perturb_set[i]);
			}
			else if (score == min_momentum) {
				cand.push_back(perturb_set[i]);
			}
			else if (score == min_momentum_2) {
				cand2.push_back(perturb_set[i]);
			}
		}
		if (cand2.size() > 0 && rand_r(&rand_seed) % 10 < 3)
			return cand2[rand_r(&rand_seed) % cand2.size()];
		return cand[rand_r(&rand_seed) % cand.size()];
	}

	int get_most_momentum_add(vector<int>& perturb_set) {
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

	void check_missing() {
		for (int i = 0; i < red_vnum; i++) {
			int temp_p = 0;
			int temp_c = 0;
			for (int j = 0; j < red_vnum - red_v_edge_cnt[i] - 1; j++) {	//no-adjacent vertices of critical node
				int second_no_adjv = no_org_v_adj_vertex[i][j];
				if (is_in_c[second_no_adjv])
					temp_p++;
				if (is_in_c[second_no_adjv] && Is_Saturated(second_no_adjv))
					temp_c++;
			}
			if (temp_p != plex_missing[i])
				cout << "temp_p != plex_missing[i]" << endl;
			if (temp_c != critical_missing[i])
				cout << "temp_c != critical_missing[i]" << endl;
		}
	}

	void check_vector() {
		for (int i = 0; i < red_vnum; i++) {
			if (is_in_c[i])
				continue;
			if (critical_missing[i] == 0 && plex_missing[i] <= param_s - 1) {
				if (!ver_exist2(vec_M1, i)) {
					cout << i << " " << " !ver_exist2(vec_M1, i)" << endl;
				}
			}
			else if (critical_missing[i] == 0 && plex_missing[i] == param_s) {
				if (!ver_exist2(vec_M2, i)) {
					cout << i << " " << " !ver_exist2(vec_M2, i)1" << endl;
				}
			}
			else if (critical_missing[i] == 1 && plex_missing[i] <= param_s) {
				if (!ver_exist2(vec_M2, i)) {
					cout << i << " " << " !ver_exist2(vec_M2, i)2" << endl;
				}
			}
		}
		for (int j = 0; j < vec_M1->vnum; j++) {
			int ver = vec_M1->vlist[j];
			if (critical_missing[ver] == 0 && plex_missing[ver] <= param_s - 1) {

			}
			else {
				cout << "M1 ELSE" << endl;
			}
		}
		for (int j = 0; j < vec_M2->vnum; j++) {
			int ver = vec_M2->vlist[j];
			if ((critical_missing[ver] == 0 && plex_missing[ver] == param_s) || (critical_missing[ver] == 1 && plex_missing[ver] <= param_s)) {

			}
			else {
				cout << "M2 ELSE" << endl;
			}
		}
	}

	void push_vertex_tabu(int vpush) {
		add_cur_vertex_new(vpush);
		time_stamp[vpush] = cycle_iter;
		/*repair*/
		int idx = 0;
		while (idx < cur_splex->vnum) {
			int vin = cur_splex->vlist[idx];
			if (Is_Overflow(vin)) {
				//printf("push_vertex_tabu\n");
				remove_cur_vertex_new(vin);
				freq[vin]++;
				freq1[vin]++;
				time_stamp[vin] = cycle_iter;
			}
			else
				idx++;
		}
	}

	void tabu_based_search(){
		ral_clear(vec_M1);
		ral_clear(vec_M2);
		cycle_iter = 0;
		int cycle_best = cur_splex->vnum;
		int fixed = -1;
		int vpush;

		for(int i = 0; i < cur_splex->vnum; i++){	//update plex_missing and critical_missing through the initial solution 
			for(int j=0; j < red_vnum-red_v_edge_cnt[cur_splex->vlist[i]]-1; j++){
				int no_adjv1 = no_org_v_adj_vertex[cur_splex->vlist[i]][j];
				//if(no_adjv1 >= red_vnum) printf("14 no_adjv1 %d red_vnum %d\n", no_adjv1, red_vnum);
				plex_missing[no_adjv1]++;
				if(plex_missing[no_adjv1] == param_s-1 && is_in_c[no_adjv1]){
					for(int k=0; k < red_vnum-red_v_edge_cnt[no_adjv1]-1; k++){
						//if(no_org_v_adj_vertex[no_adjv1][k] >= red_vnum) printf("15 no_org_v_adj_vertex %d red_vnum %d\n", no_org_v_adj_vertex[no_adjv1][k], red_vnum);
						critical_missing[no_org_v_adj_vertex[no_adjv1][k]]++;
						unadj_satu_with_M2[no_org_v_adj_vertex[no_adjv1][k]] = no_adjv1;
					}
				}
			}
		}
		for(int i = 0; i < cur_cand->vnum; i++){	//add cur_cand vertices to M(1/2/3) with the principle
			if(critical_missing[cur_cand->vlist[i]]==0 && plex_missing[cur_cand->vlist[i]]<=param_s-1) {
				ral_add(vec_M1, cur_cand->vlist[i]);
			}
			else if(critical_missing[cur_cand->vlist[i]]==0 && plex_missing[cur_cand->vlist[i]]==param_s) {
				ral_add(vec_M2, cur_cand->vlist[i]);
				M2_type[cur_cand->vlist[i]] = 1;
			}
			else if(critical_missing[cur_cand->vlist[i]] == 1 && plex_missing[cur_cand->vlist[i]] <= param_s) {
				ral_add(vec_M2, cur_cand->vlist[i]);
				M2_type[cur_cand->vlist[i]] = 0;
			}
		}
		
		while (1){
			int end = 0;
			int non_improve_iter = 0;
			while (!end){
				vector<int> vec_M1_tmp;
				for(int i = 0; i < vec_M1->vnum; ++i){ //choose (deposit>=threshold) vertices from vec_M1
					//if(deposit[vec_M1->vlist[i]] >= threshold[vec_M1->vlist[i]] || cycle_iter - time_stamp[vec_M1->vlist[i]] > 4 || cur_splex->vnum + 1 > best_size) {
					if(cycle_iter - time_stamp[vec_M1->vlist[i]] > 4 || cur_splex->vnum + 1 > best_size) {
						vec_M1_tmp.push_back(vec_M1->vlist[i]);
					}
				}
				total_add += vec_M1->vnum;
				sai_add += vec_M1_tmp.size();
				count_add++;
				if (!vec_M1_tmp.empty()){				// add vertex from vec_M1_tmp
					int vadd;
					vadd = get_most_momentum_add(vec_M1_tmp);
					ral_delete(vec_M1, vadd);
					add_cur_vertex_new(vadd);   //new add function
					time_stamp[vadd] = cycle_iter;
					if (cur_splex->vnum > cycle_best){
						cycle_best = cur_splex->vnum;
					}
					freq[vadd]++;
					freq1[vadd]++;
					vec_M1_tmp.clear();    //clear vec_M1_tmp
				}
				else{
					vector<int> vec_M2_tmp;
					for(int i = 0; i < vec_M2->vnum; ++i){ 	//choose (deposit>=threshold) vertices from vec_M2_out
						//if(deposit[vec_M2->vlist[i]] >= threshold[vec_M2->vlist[i]]) {
						if(cycle_iter - time_stamp[vec_M2->vlist[i]] > 4) {
							vec_M2_tmp.push_back(vec_M2->vlist[i]);	
						}
					}
					total_swap += vec_M2->vnum;
					sai_swap += vec_M2_tmp.size();
					count_swap++;
					if (!vec_M2_tmp.empty()){
						int pvswp;
						pvswp = get_most_momentum(vec_M2_tmp);
						int vswp_in;
						if (M2_type[pvswp] == 0){ //type 1
							if(unadj_satu_with_M2[pvswp] != -1)
								vswp_in=unadj_satu_with_M2[pvswp];
							else vswp_in = find_unadj_satu(pvswp);//pvswp������е�ĳ�����Ͷ�������
						}else{ //type 2
							vswp_in = random_unadj_with_exception(pvswp, fixed);//�ڽ��У�����pvswp���ڣ��Ҳ�Ϊfixed�Ķ���
						}
						if (flag_remove == 0 && (vswp_in == fixed || vswp_in == -1)) {
							end = 1;
						}
						else{
							remove_cur_vertex_new(vswp_in);   //new remove function
							time_stamp[vswp_in] = cycle_iter;
							add_without_change_deposit(pvswp);//SwapSet����֮�����ܱ�֤��ǰ��Ϊfeasible.
							time_stamp[pvswp] = cycle_iter;
							freq[vswp_in]++;
							freq[pvswp]++;
							freq1[vswp_in]++;
							freq1[pvswp]++;
							vec_M2_tmp.clear();		//clear vec_M2_tmp set
						}
					}
					else{
						if (flag_remove == 0) {
							vector<int> cand;
							int remove_count = 0;
							for (int i = 0; i < cur_splex->vnum; i++) {
								int ver = cur_splex->vlist[i];
								if (Is_Saturated(ver)) {
									cand.push_back(ver);
								}
							}
							while (cand.size() != 0) {
								int vvv = cand[rand_r(&rand_seed) % cand.size()];
								remove_cur_vertex_new(vvv);
								time_stamp[vvv] = cycle_iter;
								cand.clear();
								remove_count++;
								remove_flag[vvv] = 1;
								for (int i = 0; i < cur_splex->vnum; i++) {
									int ver = cur_splex->vlist[i];
									if (Is_Saturated(ver)) {
										cand.push_back(ver);
									}
								}
							}
							vector<int> ttttt;
							vector<int> ttttt2;//�Ƴ��Ķ���
							int count1 = 0;
							do {
								ttttt.clear();
								ttttt2.clear();
								for (int i = 0; i < vec_M1->vnum; ++i) { //choose (deposit>=threshold) vertices from vec_M1
									if (remove_flag[vec_M1->vlist[i]] == 0) {
										ttttt.push_back(vec_M1->vlist[i]);
									}
									else {
										ttttt2.push_back(vec_M1->vlist[i]);
									}
								}

								if (ttttt.size() != 0 && ttttt2.size() != 0) {
									if (rand_r(&rand_seed) % 10 < 5) {
										int vvv = ttttt[rand_r(&rand_seed) % ttttt.size()];
										ral_delete(vec_M1, vvv);
										add_cur_vertex_new(vvv);
										time_stamp[vvv] = cycle_iter;
									}
									else {
										int vvv = ttttt2[rand_r(&rand_seed) % ttttt2.size()];
										ral_delete(vec_M1, vvv);
										add_cur_vertex_new(vvv);
										time_stamp[vvv] = cycle_iter;
									}
									count1++;
									remove_count--;
								}
							} while (ttttt.size() != 0 && ttttt2.size() != 0);
							memset(remove_flag, 0, sizeof(int) * red_vnum);
							count_per++;
							non_improve_iter = 0;
						}
						else {
							end = 1;
						}
					}
				}
				if (cur_splex->vnum > local_best_size) {
					local_best_size = cur_splex->vnum;
					record_local_best();
				}
				if (cur_splex->vnum > best_size){
					record_best();
					if (best_size == param_best) //reach optimum
						goto ts_stop;
					non_improve_iter = 0;
				}else{
					non_improve_iter++;
				}
				if (non_improve_iter > param_s * best_size) {
					if (flag_remove == 0) {
						vector<int> m4;
						vector<int> m3;
						int max_freq = 0;
						for (int i = 0; i < red_vnum; i++) {
							if (is_in_c[i] == 1 || ver_exist2(vec_M1, i) || ver_exist2(vec_M2, i)) {
								continue;
							}
							m4.push_back(i);
							// if (critical_missing[i] == 0 && plex_missing[i] <= param_s - 1) {
							// 	cout << "critical_missing[i] == 0 && plex_missing[i] <= param_s - 1 " << restart_pass << " " << cycle_iter << " " << param_graph_file_name << endl;
							// }
							// else if (critical_missing[i] == 0 && plex_missing[i] == param_s) {
							// 	cout << "critical_missing[i] == 0 && plex_missing[i] == param_s " << restart_pass << " " << cycle_iter << " " << param_graph_file_name << endl;
							// }
							// else if (critical_missing[i] == 1 && plex_missing[i] <= param_s) {
							// 	cout << "critical_missing[cur_cand->vlist[i]] == 1 && plex_missing[cur_cand->vlist[i]] <= param_s " << restart_pass << " " << cycle_iter << " " << param_graph_file_name << endl;
							// }

							if (deposit[i] >= threshold[i] && freq1[i] > max_freq) {
								m3.clear();
								m3.push_back(i);
								max_freq = freq1[i];
							}
							else if (deposit[i] >= threshold[i] && freq1[i] == max_freq) {
								m3.push_back(i);
							}
						}
						int vpush1;
						if (m3.empty()) {
							vpush1 = m4[rand_r(&rand_seed) % m4.size()];
						}
						else {
							vpush1 = m3[rand_r(&rand_seed) % m3.size()];
						}
						freq1[vpush1] = 0;
						fixed = vpush1;
						push_vertex_tabu(vpush1);
					}
					else {
						vector<int> cand;
						int remove_count = 0;
						for (int i = 0; i < cur_splex->vnum; i++) {
							int ver = cur_splex->vlist[i];
							if (Is_Saturated(ver)) {
								cand.push_back(ver);
							}
						}
						while (cand.size() != 0) {
							int vvv = cand[rand_r(&rand_seed) % cand.size()];
							remove_cur_vertex_new(vvv);
							time_stamp[vvv] = cycle_iter;
							cand.clear();
							remove_count++;
							remove_flag[vvv] = 1;
							for (int i = 0; i < cur_splex->vnum; i++) {
								int ver = cur_splex->vlist[i];
								if (Is_Saturated(ver)) {
									cand.push_back(ver);
								}
							}
						}
						vector<int> ttttt;
						vector<int> ttttt2;//�Ƴ��Ķ���
						int count1 = 0;
						do {
							ttttt.clear();
							ttttt2.clear();
							for (int i = 0; i < vec_M1->vnum; ++i) { //choose (deposit>=threshold) vertices from vec_M1
								if (remove_flag[vec_M1->vlist[i]] == 0) {
									ttttt.push_back(vec_M1->vlist[i]);
								}
								else {
									ttttt2.push_back(vec_M1->vlist[i]);
								}
							}

							if (ttttt.size() != 0 && ttttt2.size() != 0) {
								if (rand_r(&rand_seed) % 10 < 5) {
									int vvv = ttttt[rand_r(&rand_seed) % ttttt.size()];
									ral_delete(vec_M1, vvv);
									add_cur_vertex_new(vvv);
									time_stamp[vvv] = cycle_iter;
								}
								else {
									int vvv = ttttt2[rand_r(&rand_seed) % ttttt2.size()];
									ral_delete(vec_M1, vvv);
									add_cur_vertex_new(vvv);
									time_stamp[vvv] = cycle_iter;
								}
								count1++;
								remove_count--;
							}
						} while (ttttt.size() != 0 && ttttt2.size() != 0);
						memset(remove_flag, 0, sizeof(int) * red_vnum);
						count_per++;
						non_improve_iter = 0;
					}
				}
				cur_iter++;
				cycle_iter++;
				iter_step++;
				clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time);
				totime = (double)((total_time.tv_sec - start_time.tv_sec) * 1000000 + (double)(total_time.tv_nsec - start_time.tv_nsec) / 1000) / 1000000;
				if(cycle_iter > param_cycle_iter || totime > param_max_seconds){
					goto ts_stop;
				}
			}
			if (cur_splex->vnum == 0) {
				break;
			}
			//vpush = cur_splex->vlist[rand_r(&rand_seed) % cur_splex->vnum];//����Ƴ�һ�����㡣
			//remove_cur_vertex_new(vpush);
			//time_stamp[vpush] = cycle_iter;
			if (flag_remove == 0) {
				vector<int> m4;
				vector<int> m3;
				int max_freq = 0;
				for (int i = 0; i < red_vnum; i++) {
					if (is_in_c[i] == 1 || ver_exist2(vec_M1, i) || ver_exist2(vec_M2, i)) {
						continue;
					}
					m4.push_back(i);
					// if (critical_missing[i] == 0 && plex_missing[i] <= param_s - 1) {
					// 	cout << "critical_missing[i] == 0 && plex_missing[i] <= param_s - 1 " << restart_pass << " " << cycle_iter << " " << param_graph_file_name << endl;
					// }
					// else if (critical_missing[i] == 0 && plex_missing[i] == param_s) {
					// 	cout << "critical_missing[i] == 0 && plex_missing[i] == param_s " << restart_pass << " " << cycle_iter << " " << param_graph_file_name << endl;
					// }
					// else if (critical_missing[i] == 1 && plex_missing[i] <= param_s) {
					// 	cout << "critical_missing[cur_cand->vlist[i]] == 1 && plex_missing[cur_cand->vlist[i]] <= param_s " << restart_pass << " " << cycle_iter << " " << param_graph_file_name << endl;
					// }

					if (deposit[i] >= threshold[i] && freq1[i] > max_freq) {
						m3.clear();
						m3.push_back(i);
						max_freq = freq1[i];
					}
					else if (deposit[i] >= threshold[i] && freq1[i] == max_freq) {
						m3.push_back(i);
					}
				}
				int vpush1;
				if (m3.empty()) {
					vpush1 = m4[rand_r(&rand_seed) % m4.size()];
				}
				else {
					vpush1 = m3[rand_r(&rand_seed) % m3.size()];
				}
				freq1[vpush1] = 0;
				fixed = vpush1;
				push_vertex_tabu(vpush1);
			}
			else {
				vpush = cur_splex->vlist[rand_r(&rand_seed) % cur_splex->vnum];//����Ƴ�һ�����㡣
				remove_cur_vertex_new(vpush);
				time_stamp[vpush] = cycle_iter;
			}
		}
	ts_stop:
		return;
	}

	void search_mode1 (int thread_id) {
		int reduce_num;
		fast_init_solution();
		if (cur_splex->vnum == red_vnum) goto end;
		for (int i = 0; i < red_vnum; i++) {
			momentum[i] += cur_c_deg[i];
		}
		reduce_num = is_reduced(red_min_deg);
		if (reduce_num > 0) {
			//reduce_graph(best_size - param_s);
			update_red(best_size - param_s, thread_id);
		}
		
		if (red_vnum == 0 ) goto end;
		while (1){
			restart_search();
			fast_init_solution();
			tabu_based_search();
			//update_step_in_solution();
			if (best_size == param_best)
				goto end;
			reduce_num = is_reduced(red_min_deg);
			if (reduce_num > 0){
				//reduce_graph(best_size - param_s);
				update_red(reduce_num, thread_id);
			}
			if (red_vnum <= best_size )
				goto end;
			restart_pass++;
			for (int i = 0; i < red_vnum; i++) {
				momentum[i] += local_opt_score[i];
			}
            //whether_update_group(thread_id);
			clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time);
			totime = (double)((total_time.tv_sec - start_time.tv_sec) * 1000000 + (double)(total_time.tv_nsec - start_time.tv_nsec) / 1000) / 1000000;
			if (totime > param_max_seconds ){
				break;
			}
		}
	end:
		total_start_pass = restart_pass;
		//total_time = clock();
		total_iter = cur_iter;
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &total_time);
		totime = (double)((total_time.tv_sec - start_time.tv_sec) * 1000000 + (double)(total_time.tv_nsec - start_time.tv_nsec) / 1000) / 1000000;
			
	}

	

	/*TODO:Entrance of the whole search*/
	void search_main(int thread_id){
		/*Initial data structure*/
		init_search(thread_id);
		
		search_mode1 (thread_id);
	
	}

	void report_result(){
		// cout << " Instance: " << param_graph_file_name << endl;
		// cout << " k: " << param_s << endl;
		// cout << " seed: " << param_seed << endl;
		// cout << " best solution size: " << best_size << endl;
		// cout << " best solution time: " << (float)((best_time - start_time) / CLOCKS_PER_SEC) << endl;
		// cout << " Best solution: ";
		// for (int i = 0; i < best_size; i++) {
		// 	cout << best_plex[i] << " ";
		// }

		//cout << param_graph_file_name << " k " << param_s << " seed " << param_seed << " best solution size " << best_size << " best solution time " << the_best_time << " org_vnum " << org_vnum << " red_vnum " << red_vnum << endl;
		
	}



	int check_solution(){
		int *mark = new int[org_vnum];
		memset(mark, 0, sizeof(int) * org_vnum);
		//printf("best size %d\n", best_size);
		for (int i = 0; i < best_size; i++){
			mark[best_plex[i]] = 1;
			//printf("%d\n", best_plex[i]);
		}
		//printf("\n");
		for (int i = 0; i < best_size; i++){
			int v = best_plex[i];
			int indeg = 0;
			//printf("v %d: ", v);
			for (int j = 0; j < org_v_edge_cnt[v]; j++){
				int vadj = org_v_adj_vertex[v][j];
				if (mark[vadj]) {
					indeg++;
					//printf("%d ", vadj);
				}
			}
			//printf("\n");
			if (indeg < best_size - param_s){
				printf(" v %d indeg %d best_size  %d param_s %d\n", v, indeg, best_size, param_s);
				return 0;
			}
				
		}
		delete[] mark;
		return 1;
	}
};

int isnot_load[1000 + 10];
struct ThreadData {
    int tid;
    int argc;
    char** argv;
}tid_array[100 + 10];
set<st> s1;

SplexClassic* splexclassic_solver = nullptr;

bool cmp(int a,int b)
{
	return org_decompos[red_orgid[a]] > org_decompos[red_orgid[b]];
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
						s1.erase((st){v,degree[v] + 1});   
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
		new_noadj_tbl = new int*[org_vnum];
		new_edge_count = new int[org_vnum];

		// for (int i = 0; i < org_vnum; i++) {
		// 	printf("v:%d(%d)(%d)\n", i, new_id[i], org_vnum - org_v_edge_cnt[i]);
		// 	for (int j = 0; j < org_vnum - org_v_edge_cnt[i] -1; j++)
		// 		printf("%d(%d) ", no_org_v_adj_vertex[i][j], new_id[no_org_v_adj_vertex[i][j]]);
		// 	printf("\n");
		// }

		// for (int i = 0; i < org_vnum; i++) {
		// 	printf("v:%d(%d)(%d)\n", i, new_id[i], org_v_edge_cnt[i]);
		// 	for (int j = 0; j < org_v_edge_cnt[i]; j++)
		// 		printf("%d(%d) ", org_v_adj_vertex[i][j], new_id[org_v_adj_vertex[i][j]]);
		// 	printf("\n");
		// }
		
		for (int idx_prev = 0; idx_prev < org_vnum; idx_prev++)
		{
			int idx_new = new_id[idx_prev];
			new_adj_tbl[idx_new] = new int[org_v_edge_cnt[idx_prev]];
			new_noadj_tbl[idx_new] = new int[org_vnum-org_v_edge_cnt[idx_prev]-1];
			new_edge_count[idx_new] = org_v_edge_cnt[idx_prev];  // 把原来的顶点度赋给新编号对应的顶点度
			int cnt = 0;
			for (int i = 0; i < org_v_edge_cnt[idx_prev]; i++) {
				int vi_adj = red_v_adj_vertex[idx_prev][i];
				new_adj_tbl[idx_new][cnt++] = new_id[vi_adj];
			}
			sort(new_adj_tbl[idx_new],new_adj_tbl[idx_new] + new_edge_count[idx_new],cmp);
			
			
			cnt = 0;
			for(int i = 0; i < org_vnum - org_v_edge_cnt[idx_prev] - 1; i++) {
				int no_adjv = no_org_v_adj_vertex[idx_prev][i];
				new_noadj_tbl[idx_new][cnt++] = new_id[no_adjv];
			}
			sort(new_noadj_tbl[idx_new],new_noadj_tbl[idx_new] + (org_vnum - new_edge_count[idx_new] - 1),cmp);
			
		}
		
		/*for (int i = 0; i < org_vnum; i++) {
			printf("%d(%d):\n", i, org_decompos[red_orgid[i]]);
			for(int j = 0; j < org_vnum - org_v_edge_cnt[i] - 1; j++) {
				int no_adjv = no_org_v_adj_vertex[i][j];
				printf("%d(%d) ", no_adjv, org_decompos[red_orgid[no_adjv]]);
			}
			printf("\n");
		}*/
		
	
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
		//delete[] no_org_v_adj_vertex;
		//org_v_edge_cnt = new_edge_count;
		red_v_adj_vertex = new_adj_tbl;
		no_org_v_adj_vertex = new_noadj_tbl;
		for(int i = 0;i < thread_size;i++) 
		{
			splexclassic_solver[i].red_vnum = rem; //org_vnum;
			splexclassic_solver[i].red_enum = org_enum;
			splexclassic_solver[i].red_v_edge_cnt = new int[org_vnum];
			memcpy(splexclassic_solver[i].red_v_edge_cnt, new_edge_count, sizeof(int) * org_vnum);
		}
		
		// for (int idx = 0; idx < org_vnum; idx++)
		// {
		// 	printf("new:%d(%d)\n", idx, org_vnum - org_v_edge_cnt[idx]);
		// 	for(int i = 0; i < org_vnum - org_v_edge_cnt[idx] - 1; i++) {
		// 		int no_adjv = no_org_v_adj_vertex[idx][i];
		// 		printf("%d ", no_adjv);
		// 	}
		// 	printf("\n");
		// }

		// for (int idx = 0; idx < org_vnum; idx++)
		// {
		// 	printf("new:%d(%d)\n", idx, new_edge_count[idx]);
		// 	for(int i = 0; i < new_edge_count[idx]; i++) {
		// 		int adjv = red_v_adj_vertex[idx][i];
		// 		printf("%d ", adjv);
		// 	}
		// 	printf("\n");
		// }
		

		// for (int i = 0; i < org_vnum; i++) {
		// 	printf("old %d new %d layer %d\n", i, new_id[i], v_in_layer[new_id[i]]);
		// }
	

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
		low_group_size = (sum_v / v_group.size()) * gama1;
	    up_group_size = (sum_v / v_group.size()) * gama2;
	}
    

	/*printf("init size %ld  group vnum %d total vnum %d raito %lf\n", v_group.size(), sum_v, org_vnum, (sum_v*1.0)/org_vnum);
    for (int i = 0; i < v_group.size(); i++) {
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


void *create_search(void *arg)
{
	ThreadData *data = (ThreadData*)arg;
	int i = data->tid;

	if(isnot_load[i]) return NULL;
	splexclassic_solver[i].search_main(i);
	return NULL;
}

int main(int argc, char** argv) {
	load_instance(argc,argv);
	//SplexClassic splexclassic_solver[thread_size];
	splexclassic_solver = new SplexClassic[thread_size];
	for(int i = 0;i < thread_size;i++) splexclassic_solver[i] = SplexClassic();
	for(unsigned int i = 1;i <= thread_size;i++) splexclassic_solver[i].set_seed(i);
	global_init_search();
	//group_partition();

	pthread_t *ptr = new pthread_t[thread_size];
	for(int tid = 0;tid < thread_size;tid++) 
	{
		tid_array[tid].tid = tid;
		tid_array[tid].argc = argc;
		tid_array[tid].argv = argv;
		pthread_create(&ptr[tid], NULL, create_search,(void*)&tid_array[tid]);
	}
	for(int tid = 0;tid < thread_size;tid++) pthread_join(ptr[tid], NULL);

    for(int i = 0;i < thread_size;i++)
	{
		int chk = splexclassic_solver[i].check_solution();
		
		if (chk == 0){
			//printf("ERROR! Final solution is infeasible at thread %d\n",i);
			printf("ERROR! %d %s Final solution is infeasible at thread %d\n",param_s,param_graph_file_name,i);
			//exit(0);
		}
		else splexclassic_solver[i].report_result();
		gloabl_avg += splexclassic_solver[i].best_size;
		
		//gloabl_avg += splex_solver[i].best_size;
		//splex_solver[i].free_memory();
	}
	//cout << "final " << param_graph_file_name << " " << param_s << " " << global_best_size << " " << (double)gloabl_avg / thread_size << " " << global_best_time << endl;
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

}
