#include<bits/stdc++.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace std;
using namespace OpenNN;

string convert_to_string(int n)
{
      string s;
      while(n)
      {
            s.push_back(n % 10 + '0');
            n /= 10;
      }

      reverse(s.begin(), s.end());
      return s;
}

int main(void)
{
   try
   {
      std::cout << "PSE Load prediction training" << std::endl;

      srand((unsigned)time(NULL));

      // Data set

      DataSet data_set;

      data_set.set_data_file_name("load_data.dat");

      data_set.set_separator("Space");

      data_set.load_data();

      // Variables

      Variables* variables_pointer = data_set.get_variables_pointer();

      Vector< Variables::Item > variables_items(51);

      variables_items[0].name = "year";
      variables_items[0].use = Variables::Input;

      variables_items[1].name = "month";
      variables_items[1].use = Variables::Input;

      variables_items[2].name = "day";
      variables_items[2].use = Variables::Input;

      for(int i = 3; i < 27; i++)
      {
            variables_items[i].name = "temp_hour" + convert_to_string(i - 2);
            variables_items[i].units = "degrees";
            variables_items[i].use = Variables::Input;
      }

      for(int i = 27; i < 51; i++)
      {
            variables_items[i].name = "load_hour" + convert_to_string(i - 26);
            variables_items[i].units = "Watts";
            variables_items[i].use = Variables::Target;

      }

      variables_pointer->set_items(variables_items);

      const Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
      const Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

      // Instances

      Instances* instances_pointer = data_set.get_instances_pointer();

      instances_pointer->split_random_indices();

      const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();
      const Vector< Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();

      // Neural network

      NeuralNetwork neural_network(27, 10, 24);

      Inputs* inputs = neural_network.get_inputs_pointer();

      inputs->set_information(inputs_information);

      Outputs* outputs = neural_network.get_outputs_pointer();

      outputs->set_information(targets_information);

      neural_network.construct_scaling_layer();

      ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

      scaling_layer_pointer->set_statistics(inputs_statistics);

      scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);

      neural_network.construct_unscaling_layer();

      UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();

      unscaling_layer_pointer->set_statistics(targets_statistics);

      unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);

      // Performance functional

      PerformanceFunctional performance_functional(&neural_network, &data_set);

      performance_functional.set_regularization_type(PerformanceFunctional::NEURAL_PARAMETERS_NORM);

      // Training strategy object

      TrainingStrategy training_strategy(&performance_functional);

      QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

      quasi_Newton_method_pointer->set_maximum_iterations_number(1000);
      quasi_Newton_method_pointer->set_display_period(10);

      quasi_Newton_method_pointer->set_minimum_performance_increase(1.0e-6);

      quasi_Newton_method_pointer->set_reserve_performance_history(true);

      TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

      // Testing analysis

      TestingAnalysis testing_analysis(&neural_network, &data_set);

      TestingAnalysis::LinearRegressionResults linear_regression_results = testing_analysis.perform_linear_regression_analysis();

      // Save results

      scaling_layer_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);
      unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::MinimumMaximum);

      data_set.save("data_set.xml");

      neural_network.save("neural_network.xml");
      neural_network.save_expression("expression.txt");

      performance_functional.save("performance_functional.xml");

      training_strategy.save("training_strategy.xml");
      training_strategy_results.save("training_strategy_results.dat");

      linear_regression_results.save("linear_regression_analysis_results.dat");
	
      //Input to output here

      Vector<double> inp (27);

	while(1)
	{
            std::cout<<"Enter the 5 values here\n";
            for(int i = 0; i < 27; i++)
                  cin>>inp[i];
		Vector<double> out = neural_network.calculate_outputs(inp);

            cout<<"The hourly output is : \n";
            for(int i = 0; i < out.size(); i++)
                  cout<<out[i]<<" ";
            std::cout<<"\n";
	}
	
      return(0);
   }
   catch(std::exception& e)
   {
      std::cerr << e.what() << std::endl;

      return(1);
   }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2015 Roberto Lopez.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
