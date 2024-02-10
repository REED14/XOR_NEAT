#include <iostream>
#include <algorithm>
#include <vector>
#include <stack>
#include <random>
#include <cmath>
#include <fstream>

using namespace std;

struct conn{
	pair<int, int> ids;
	double weight;
};

struct node{
	vector<pair<int, double>> inputs;
	bool computed;
	double value;
};

/*
- min_nc - Minimum new connections per cycle
- max_nc - Maximum new connections per cycle

- min_wv - Minimum weight value
- max_wv - Maximum weight value

- in_num - number of inputs
- out-num - number of outputs

- hidden - number of hidden nodes

*/

double sigmoid(double x){
	return 1 / 1 + exp(-1.0*x);
}

class NEAT{
private:
	double fitness;
	int inputs_num, outputs_num, hidden_num;
	bool is_computable;
	vector<int> output_ids;
	vector<node> nodes;
	vector<conn> conns;

public:
	NEAT(int min_nc, int max_nc, double min_wv, double max_wv, int in_num, int out_num, int hidden, vector<conn> connects){
		//initializing the input nodes
		inputs_num = in_num;
		for (int i = 1; i <= in_num; ++i)
			nodes.push_back({ vector<pair<int, double>>(), true, 0 });

		//initializing output nodes
		outputs_num = out_num;
		for (int i = 1; i <= out_num; ++i){
			output_ids.push_back(nodes.size());
			nodes.push_back({ vector<pair<int, double>>(), false, 0 });
		}

		//initializing hidden nodes
		hidden_num = hidden;
		for (int i = 1; i <= hidden; ++i){
			random_device rd;
			mt19937 gen(rd());
			uniform_real_distribution<> biasd(-1.0, 1.0);
			nodes.push_back({ vector<pair<int, double>>(), false, 0 });
		}
		
		//adding connections to every node if connections exist exist
		conns = connects;
		for (auto const &x : conns){
			//if (x.ids.second > nodes.size()-1) continue;
			nodes[x.ids.second].inputs.push_back({x.ids.first, x.weight});
		}

		//adding connections randomly if necessary
		if (min_nc > max_nc || max_wv < min_wv) is_computable = 0;
		else if(max_nc > 0){
			random_device rd;
			mt19937 gen(rd());
			uniform_int_distribution<> distr(min_nc, max_nc);

			int total_new_cons = distr(gen);
			for (int i = 0; i <= total_new_cons; ++i)
				gen_connection(in_num, out_num, min_wv, max_wv);
		}

		//print_cons();
		//gen_node_on_connection(conns[0], 1);
		//calculate_output(2);
	}

	int getNodesNumber(){
		return nodes.size();
	}

	int getHiddenNodesNumber(){
		return hidden_num;
	}

	vector<conn> getConnections(){
		return conns;
	}

	//generate weight randomly
	double gen_weight(double minw, double maxw){
		static random_device rd;
		static mt19937 gen(rd());
		uniform_real_distribution<> wei(minw, maxw);

		return wei(gen);
	}

	//generate connection randomly
	conn gen_connection_random(double min_wv, double max_wv){
		static random_device rd;
		static mt19937 gen(rd());

		//selecting random nodes
		conn ncn;
		uniform_int_distribution<> conn_node(0, nodes.size() - 1);
		uniform_real_distribution<> conn_wei(min_wv, max_wv);
		ncn.ids.second = conn_node(gen);
		ncn.ids.first = conn_node(gen);
		ncn.weight = gen_weight(min_wv, max_wv);

		return ncn;
	}

	//verify if connection is recursive
	bool is_connection_recursive(conn ncn){
		bool is_recursive = 0;

		if (ncn.ids.first == ncn.ids.second)
			return 1;

		vector<pair<int, double>> to_iter(nodes[ncn.ids.first].inputs), next_iter;
		while (is_recursive == 0 && to_iter.size()>0){
			for (auto const &x : to_iter){
				if (x.first == ncn.ids.first || x.first == ncn.ids.second) 
					return true;
				for (auto const &ins : nodes[x.first].inputs)
					next_iter.push_back(ins);
			}
			to_iter.clear();
			to_iter = next_iter;
			next_iter.clear();
		}

		return 0;
	}

	//generate connections
	void gen_connection(int in_num, int out_num, double min_wv, double max_wv){
		conn ncn = gen_connection_random(min_wv, max_wv);

		//checking if nodes are structurally suitable
		bool is_first_output = 0, is_second_input = 0, is_recursive = 0, is_same = 0, is_ok = 0;
		int iters = 0;

		//if the nodes are not ok it will generate other connections (it will try for a maximum of 100 times to generate a new connection)
		while (iters < 100){
			for (auto const &x : output_ids)
				if (ncn.ids.first == x) is_first_output = true;
			if (ncn.ids.second < in_num) is_second_input = true;

			//checking if connection already exists
			for (auto const &x : nodes[ncn.ids.second].inputs)
			if (x.first == ncn.ids.first) { is_same = 1; break; }

			//check if first node and second node are the same
			if (ncn.ids.first == ncn.ids.second) is_same = 1;

			//checking nodes create a recursive loop
			is_recursive = is_connection_recursive(ncn);
			iters++;

			//if it's ok the while loop will stop and the connection will be added
			if (is_recursive == 0 && is_first_output == 0 && is_second_input == 0 && is_same == 0)
			{
				nodes[ncn.ids.second].inputs.push_back({ ncn.ids.first, ncn.weight }); 
				conns.push_back(ncn); 
				is_ok = 1;
				break;
			}

			//otherwhise everything will be reset and a new connection will be generated
			else{
				is_first_output = is_second_input = is_recursive = is_same = 0;
				ncn = gen_connection_random(min_wv, max_wv);
			}
		}
	}

	//generates node on connection
	void gen_node_on_connection(conn connection, double weight){
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<> biasd(-1.0, 1.0);

		//create node
		node NewNode;
		NewNode.value = 0;
		NewNode.computed = 0;
		NewNode.inputs.push_back({ connection.ids.first, weight });
		nodes.push_back(NewNode);

		//removes connection input data to first node inside second node
		nodes[connection.ids.second].inputs.erase(
			remove_if(nodes[connection.ids.second].inputs.begin(), nodes[connection.ids.second].inputs.end(), 
			[connection](pair<int, double> val) {return (connection.ids.first == val.first); }),
		nodes[connection.ids.second].inputs.end());

		//add new node to inputs with assigned weight
		nodes[connection.ids.second].inputs.push_back({ nodes.size() - 1 , connection.weight});

		//removes connection from genome's connections vector
		conns.erase(remove_if(conns.begin(), conns.end(), 
			[connection](conn val){return (connection.ids.first == val.ids.first && connection.ids.second == val.ids.second); }), 
		conns.end());

		//adds the 2 connections to connection vector
		conns.push_back({ { connection.ids.first, nodes.size() - 1 }, weight });
		conns.push_back({ { nodes.size() - 1 , connection.ids.second }, connection.weight });

		hidden_num++;
	}

	//calculate output
	void calculate_output(const int out_id){
		//verifying if function's parameters are appropriate
		if (!count(output_ids.begin(), output_ids.end(), out_id)){
			cout << "Id is not in the outputs!\n\n"; 
			return;
		}
		
		//setting up the stack for graph iteration
		stack<pair<int, int>> ids_to_compute;
		ids_to_compute.push({ out_id , 0 });

		//just for good measure to prevent bugs
		for (auto const &x : conns){
			//if (x.ids.second > nodes.size()-1) continue;
			bool add_connection = 1;

			for (auto &y:nodes[x.ids.second].inputs)
			if (y.first == x.ids.first) add_connection = 0;

			if (add_connection)
				nodes[x.ids.second].inputs.push_back({ x.ids.first, x.weight });
		}

		//network/graph itteration/computation
		while (!ids_to_compute.empty()){

			int current_id, current_input;
			current_id = ids_to_compute.top().first; 
			current_input = ids_to_compute.top().second;

			//cout << current_id << " " << current_input << " " << nodes[current_id].inputs.size() << '\n';
			if (nodes[current_id].inputs.size() == 0 && nodes[current_id].computed == 0)
				break;
			for (int i = current_input; i < nodes[current_id].inputs.size(); ++i){
				
				int node_to_add = nodes[current_id].inputs[i].first;
				double node_to_add_weight = nodes[current_id].inputs[i].second;

				//for debugging purposes only
				//cout << current_id << " <- " << node_to_add << '\n';

				if (nodes[node_to_add].computed) {
					//cout << "sugi_pula\n";
					nodes[current_id].value += 1.0 * nodes[node_to_add].value * node_to_add_weight;
					
					if (i == nodes[current_id].inputs.size() - 1){
						nodes[current_id].computed = true;
						//ADDING SIGMOID
						if (!count(output_ids.begin(), output_ids.end(), current_id))
							nodes[current_id].value = sigmoid(nodes[current_id].value);
						else{
							if (nodes[current_id].value > 0.5) nodes[current_id].value = 1;
							else nodes[current_id].value = 0;
						}
					}
				}
				else{
					ids_to_compute.pop();
					ids_to_compute.push({ current_id, i });
					ids_to_compute.push({ node_to_add, 0 });
					break;
				}
			}
			if (nodes[current_id].computed) ids_to_compute.pop();
		}

		//nodes[out_id].value = sigmoid(nodes[out_id].value);
	}

	void reset_network(){
		for (auto &x : nodes){
			x.computed = 0;
			x.value = 0;
			//cout << x.computed << '\n';
		}
	}

	//calculating on given inputs
	void compute_network(vector<double> inputs){
		//check if inputs vector is correct
		if (inputs.size() != inputs_num)
		{
			cout << "Inadequete inputs vector";
			return;
		}
		
		//resetting neural network
		reset_network();

		//setting up inputs
		for (int i = 0; i < inputs_num; ++i){
			nodes[i].value = inputs[i];
			nodes[i].computed = 1;
		}

		//computing outputs
		for (auto &output : output_ids){
			calculate_output(output);
			//cout << nodes[output].value << " ";
		}
	}

	//print result
	void showResult(vector<double> inputs){
		compute_network(inputs);
		cout << "results: ";
		for (auto &x : output_ids)
			cout << nodes[x].value << " ";
		cout << '\n';
	}

	//fitness function(for this problem)
	void computeFitness(const vector<pair<vector<double>, vector<double>>> data_batch){
		double mistake_sum = 0;
		fitness = 0;
		for (auto const &data : data_batch){
			//check if inputs vector is correct
			if (data.first.size() != inputs_num)
			{
				cout << "Inadequete inputs vector";
				return;
			}
			//check if outputs vector is correct
			if (data.second.size() != outputs_num)
			{
				cout << "Inadequete outputs vector";
				return;
			}

			compute_network(data.first);

			int outputs_iter = 0;
			for (auto const &x : output_ids){
				mistake_sum += (nodes[x].value - data.second[outputs_iter])*(nodes[x].value - data.second[outputs_iter]);
				++outputs_iter;
			}
		}

		fitness = 100 / (1 + mistake_sum);
		//fitness = 1/exp(mistake_sum);
	}

	double getFitness(){
		return fitness;
	}

	//remove connection randomly
	void rmconnection(){
		static random_device rd;
		static mt19937 gen(rd());
		uniform_int_distribution<> rmdist(0, conns.size()-1);

		int conntoremidx = rmdist(gen);
		conn conntorem = conns[conntoremidx];
		int id_to_rem = -1, idx=0;
		for (auto &x : nodes[conntorem.ids.second].inputs){
			if (x.first == conntorem.ids.first) { id_to_rem = idx; break; }
			++idx;
		}

		if (id_to_rem != -1)
			nodes[conntorem.ids.second].inputs.erase(nodes[conntorem.ids.second].inputs.begin() + id_to_rem);

		conns.erase(conns.begin() + conntoremidx);
	}

	//mutate the genome
	void mutate(int mut_max_conns, double rmconchance, double addconchance, double addnodechance, double min_max_wv, double min_max_wv_mut){
		//setting up random device
		static random_device rd;
		static mt19937 gen(rd());

		//setting up distributions
		uniform_real_distribution<> chance(0, 1);
		uniform_real_distribution<> nwvd((-1.0)*min_max_wv, min_max_wv);
		uniform_real_distribution<> mwvd((-1.0)*min_max_wv_mut, min_max_wv_mut);

		//mutating weights randomly
		uniform_int_distribution<> weight_to_mut(0, mut_max_conns);
		for (int i = 0; i < weight_to_mut(gen); ++i){
			uniform_int_distribution<> conn_to_choose(0, conns.size() - 1); 
			conns[conn_to_choose(gen)].weight += mwvd(gen);
		}

		//removing connection if chance is given
		if (chance(gen) <= rmconchance) rmconnection();

		//adding connection if chance is given
		if (chance(gen) <= addconchance)
			gen_connection(inputs_num, outputs_num, (-1.0)*min_max_wv, min_max_wv);

		//adding node if necessary
		if (chance(gen) <= addnodechance){
			uniform_int_distribution<> conn_to_choose(0, conns.size() - 1);
			if (conns.size()>conn_to_choose(gen))
			gen_node_on_connection(conns[conn_to_choose(gen)], nwvd(gen));
		}
	}

	void print_structure(){
		for (auto const &x : conns)
			cout << x.ids.first << "->" << x.ids.second << " " << x.weight << '\n';
	}
};

class Population{
private:
	int genomes;
	double minimum_dissimilarity;
	vector<pair<vector<double>, vector<double>>> dataset;
	vector<NEAT> currentGenomes;
	int min_nc, max_nc;
	double min_wv, max_wv;

	//mutation variables
	int mut_max_conns; 
	double rmconchance, addconchance, addnodechance, min_max_wv, min_max_wv_mut;
		
public:
	Population(int sizePop, int minnc, int maxnc, double minwv, double maxwv, double min_dis, vector<pair<vector<double>, vector<double>>> data,
			   int imut_max_conns, double irmconchance, double iaddconchance, double iaddnodechance, double imin_max_wv, double imin_max_wv_mut)
	{
		//setting up the basic data
		dataset = data;
		genomes = sizePop;

		min_nc = minnc; max_nc = maxnc;
		min_wv = minwv; max_wv = maxwv;

		minimum_dissimilarity = min_dis;

		//initializing population
		for (int i = 0; i < genomes; ++i){
			NEAT genome(min_nc, max_nc, min_wv, max_wv, dataset[0].first.size(), dataset[0].second.size(), 0, vector<conn>());
			currentGenomes.push_back(genome);
		}

		mut_max_conns = imut_max_conns;
		rmconchance = irmconchance; addconchance = iaddconchance; addnodechance = iaddconchance;
		min_max_wv = imin_max_wv; min_max_wv_mut = imin_max_wv_mut;
	}

	//performing computations for every genome
	void compute_population_fitness(){
		for (auto &x : currentGenomes)
			x.computeFitness(dataset);
	}

	//computing dissimilarity between 2 NEAT models
	double compute_dissimilarity(NEAT& const n1,NEAT& const n2){
		double disim_factor = 0;
		vector<conn> c1 = n1.getConnections();
		vector<conn> c2 = n2.getConnections();

		//parsing c1 connections
		for (auto const &x : c1){
			double c2w = 0;
			for (auto const &y : c2){
				if (x.ids.first == y.ids.first && x.ids.second == y.ids.second){
					c2w = y.weight;
					break;
				}
			}
			disim_factor += abs(x.weight - c2w);
		}

		//parsing c2 connection for inputs not taken into consideration
		for (auto const &y : c2){
			double c2w = 0;
			for (auto const &x : c1){
				if (x.ids.first == y.ids.first && x.ids.second == y.ids.second){
					c2w = y.weight;
					break;
				}
			}
			if (c2w == 0)
				disim_factor += abs(y.weight);
		}

		return disim_factor;
	}

	//generating species
	vector<vector<NEAT>> species(){
		vector<vector<NEAT>> result;
		vector<NEAT> genomesCopy = currentGenomes;

		int vec = 0;
		while (!genomesCopy.empty()){
			//putting vectors where they should be in speciation
			NEAT primordial = genomesCopy[0];
			result.push_back({ primordial });

			//comparing to the rest of specimens
			stack<int> genome_to_remove;
			int idx = 0;
			for (auto &x : genomesCopy){
				if (compute_dissimilarity(primordial, x) < minimum_dissimilarity)
				{
					result[vec].push_back(x);
					genome_to_remove.push(idx);
				}
				++idx;
			}

			//removing elements in species
			while (!genome_to_remove.empty()){
				genomesCopy.erase(genomesCopy.begin() + genome_to_remove.top());
				genome_to_remove.pop();
			}
			++vec;
		}
		return result;
	}

	//crossover
	NEAT Crossover(NEAT& const n1, NEAT& const n2){
		vector<conn> c1 = n1.getConnections();
		vector<conn> c2 = n2.getConnections();

		vector<conn> c3;
		for (auto x : c1){
			vector<conn>::iterator to_add = find_if(c2.begin(), c2.end(), [&x](const conn& c){
				return c.ids == x.ids;
			});
			if (to_add != c2.end())
				c3.push_back({ (*to_add).ids, ((*to_add).weight + x.weight) / 2 });
			else
				c3.push_back(x);
		}

		return NEAT(0, 0, min_wv, max_wv, dataset[0].first.size(), dataset[0].second.size(), n1.getHiddenNodesNumber(), c3);
	}

	//generate genomes from vector
	NEAT offspring(vector<NEAT> specie){
		static random_device rd;
		static mt19937 gen(rd());

		pair<int, int> parents;
		uniform_int_distribution<> p1(0, specie.size()/2);
		parents.first = p1(gen);

		uniform_int_distribution<> p2(parents.first, specie.size()-1);
		parents.first = p2(gen);

		NEAT result = Crossover(specie[parents.first], specie[parents.second]);
		result.mutate(mut_max_conns, rmconchance, addconchance, addnodechance, min_max_wv, min_max_wv_mut);
		return result;
	}

	//creating new generation of genomes
	void NewGen(){
		struct SortGenomeByFitness {
			bool operator()(NEAT& const n1, NEAT& const n2){
				if (n1.getFitness() > n2.getFitness())
					return true;
				return false;
			}
		};

		vector<vector<NEAT>> specs = species();
		
		//get average fitness
		double avgFitness = 0; int popSize = 0;
		for (auto &spec : specs)
			for (auto &x : spec)
			{
				avgFitness += x.getFitness(); ++popSize;
			}
		avgFitness /= 1.0*popSize;

		//removing unfit population
		for (auto &spec : specs){
			sort(spec.begin(), spec.end(), SortGenomeByFitness());
			if (spec.size() == 1 && spec[0].getFitness() >= avgFitness)
				continue;
			spec.erase(spec.begin() + spec.size() / 2, spec.end());
		}
		
		/*
		sort(currentGenomes.begin(), currentGenomes.end(), SortGenomeByFitness());
		currentGenomes.erase(currentGenomes.begin() + currentGenomes.size() / 2, currentGenomes.end());
		for (auto &spec : currentGenomes){
			
			if (spec.size() == 1 && spec[0].getFitness() >= avgFitness)
				continue;
		
		}

		//repopulating
		vector<NEAT> repGenomes = move(currentGenomes);
		currentGenomes.clear();
		for (int i = 0; i < genomes; ++i)
			currentGenomes.push_back(offspring(repGenomes));*/

		//repGenomes.clear();

		//repopulating
		currentGenomes.clear();
		for (int i = 0; i < genomes; ++i)
			currentGenomes.push_back(offspring(specs[i%(specs.size()-1)]));
	}

	NEAT getFitest(){

		int hasBiggestFitness = 0;
		double biggestFitness = 0;
		int idx = 0;
		for (auto &x : currentGenomes){
			//x.print_structure();
			//cout << '\n';

			x.computeFitness(dataset);
			double currentfit = x.getFitness();
			if (x.getFitness() > biggestFitness)
				biggestFitness = currentfit, hasBiggestFitness = idx;
			++idx;
		}

		return currentGenomes[hasBiggestFitness];
	}
	
	//this is only for XOR
	void showDataOfBest(){
		NEAT fitest = getFitest();
		cout << "Best Network:\n Structure:\n";
		fitest.print_structure();
		cout << "Fitness: "<<fitest.getFitness()<<'\n';
		cout << "Dataset:\n";
		for (auto &x : dataset){
			cout << "\tinputs : {";
			for (auto &input : x.first)
				cout << input <<", ";
			cout << "\b\b}\n\t";
			fitest.showResult(x.first);
		}
		cout << '\n';
	}
};


int main(){
	//NEAT(int min_nc, int max_nc, int min_wv, int max_wv, int in_num, int out_num, int hidden, vector<conn> connects)
	//NEAT test(1, 2, -2.2, 2.2, 2, 1, 0, vector<conn>());
	//cout << '\n';
	//test.print_cons();

	//cout << "Computing for [1, 1]\n\n";
	//test.compute_network({1, 1});

	const vector<pair<vector<double>, vector<double>>> dataset = {
		{ { 0, 1 }, { 1 } },
		{ { 1, 0 }, { 1 } },
		{ { 1, 1 }, { 0 } }
	};

	//hoping for the best 
	Population p(300, 1, 3, -2, 2, 0.5, dataset, 10, 0.05, 0.3, 0.15, 1, 0.05);
	int iters = 1000;
	for (int i = 0; i < iters; i++){
		p.showDataOfBest();
		p.NewGen();
	}



	//doing the fittness for xor
	//cout << test.fitness(dataset) << '\n';

	cin.get();
	return 0;
}