/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "archs/Astra.h"
#include "misc/config.h"
#include "numerical/finite_difference.h"
#include "trainer.h"
#include "quantitize.h"

#include "dataset/dataset.h"
#include "dataset/writer.h"
#include "position/fenparsing.h"

#include <iostream>
#include <vector>
#include <filesystem> 

using namespace std;
namespace fs = std::filesystem;

vector<string> loadCsv(const std::string& filename) {
    vector<std::string> fenEvalPairs;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return fenEvalPairs; 
    }

    string line;
    while (getline(file, line)) {
        istringstream ss(line);
        string fen, eval;

        if (getline(ss, fen, ',') && getline(ss, eval)) {
            fenEvalPairs.push_back(fen + " " + eval);
        }
    }

    file.close();
    return fenEvalPairs;
}

int main() {
    init();
    
    /*
    const string output_folder = "C:/Users/semio/Documents/programming/Astra-Chess-Engine/Astra-Data/bin/";
    
    // validation data
    const string val_data_path = "C:/Users/semio/Documents/programming/Astra-Chess-Engine/Astra-Data/val_data/chess_val_data1_d9.csv";

    vector<string> val_data = loadCsv(val_data_path);
    cout << "Loaded " << val_data.size() << " val positions" << std::endl;

    vector<Position> val_positions;
    for (const string& s : val_data) {
        val_positions.push_back(parseFen(s));
    }

    DataSet val_dataset;
    val_dataset.positions = val_positions;

    write(output_folder + "val_data.bin", val_dataset, val_dataset.positions.size());

    // training data
    const string training_data_folder  = "C:/Users/semio/Documents/programming/Astra-Chess-Engine/Astra-Data/training_data/";

    int index = 1;
    for (const auto& entry : fs::directory_iterator(training_data_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            string filePath = entry.path().string();
            vector<string> training_data = loadCsv(filePath);
            cout << "Loaded " << training_data.size() << " training positions from " << filePath << endl;

            vector<Position> training_positions;
            for (const string& s : training_data) {
                training_positions.push_back(parseFen(s));
            }

            DataSet training_dataset;
            training_dataset.positions = training_positions;

            write(output_folder + to_string(index) + ".bin", training_dataset, training_dataset.positions.size());
            index++;
        }
    }

    return 0;
    */

    const string data_path = "C:/Users/semio/Documents/programming/Astra-Chess-Engine/Astra-Data/bin/";
    const string output    = "C:/Users/semio/Documents/programming/Astra-Chess-Engine/Astra-Data/nn_output/";
    /*
    vector<string> files {};
    for (int i = 1; i <= 1; i++) {
        files.push_back(data_path + to_string(i) + ".bin");
    }

    Trainer<Astra> trainer {};
    trainer.fit(files, vector<string> {data_path + "val_data.bin"}, output);
*/
    auto layers = Astra::get_layers();
    Network network{std::get<0>(layers),std::get<1>(layers)};
    network.setLossFunction(Astra::get_loss_function());
    network.loadWeights(output + "weights-epoch20.nnue");

    test_fen<Astra>(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    test_fen<Astra>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");

    quantitize_shallow(output + "nn-768-2x256-1.nnue", network);

    std::cout << "end" << std::endl;

    close();
}
