#include<bits/stdc++.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;

int main(void)
{
   try
   {
      cout << "PSE Assignment (Load Prediction)\n";

      NeuralNetwork neural_network;
      neural_network.load("neural_network.xml");

      Vector<double> inp (27);

	while(1)
	{
            cout<<"Enter the 27 values here\n";
            for(int i = 0; i < 27; i++)
                  cin>>inp[i];
		Vector<double> out = neural_network.calculate_outputs(inp);

            cout<<"The hourly load output is : \n";
            for(int i = 0; i < out.size(); i++)
                  cout<<out[i]<<" ";
            cout<<"\n";
	}
	
      return(0);
   }
   catch(std::exception& e)
   {
      std::cerr << e.what() << std::endl;

      return(1);
   }
}
