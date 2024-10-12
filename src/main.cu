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

using namespace std;

vector<string> loadCsv(const std::string& filename) {
    vector<std::string> fenEvalPairs;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return fenEvalPairs; // Return empty vector on failure
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
    const string val_data_path = "C:/Users/semio/Documents/programming/Astra-Chess-Engine/Astra-Data/val_data/chess_val_data1_d9.csv";

    vector<string> val_data = loadCsv(val_data_path);

    cout << "Loaded " << val_data.size() << " positions" << std::endl;

    vector<Position> positions;
    for (const string& s : val_data) {
        positions.push_back(parseFen(s));
        if(positions.size() == 1000) break;
    }

    DataSet dataset;

    dataset.positions = positions;

    string test = "C:/Users/semio/Downloads/test.bin";

    write(test, dataset, dataset.positions.size());
    return 0;
    */

    string test = "C:/Users/semio/Downloads/test.bin";

    const string data_path = "C:/Users/semio/Documents/programming/Astra-Chess-Engine/Astra-Data/training_data/";
    const string output    = "C:/Users/semio/Documents/programming/Astra-Chess-Engine/Astra-Data/nn_output/";
    
    vector<string> files {test};
    //for (int i = 1; i <= 200; i++) {
      //  files.push_back(data_path + to_string(i) + ".bin");
    //}

    Trainer<Astra> trainer {};
    trainer.fit(files, vector<string> {test}, output);

    close();
}
