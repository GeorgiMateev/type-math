using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TypeMath.NeuralNetwork;

namespace TypeMath.Cmd
{
    public class IrisData
    {
        public void Learn()
        {
            var trainDataFileName = "iris_train.txt";
            var data = IrisData.ReadData(trainDataFileName);

            var hiddenLayerNeurons = 20;
            var iterations = 500;
            var learningConst = 0.1;
            var net = new Network(4, 3, hiddenLayerNeurons);
            net.Train(data, iterations, learningConst);

            var testFileName = "iris_test.txt";
            var resultsFileName = "iris_test_result.txt";

            var testCases = IrisData.ReadData(testFileName);
            var testResults = IrisData.ReadData(resultsFileName).Select(x => x.First()).ToList();

            var result = net.Classify(testCases);

            for (int i = 0; i < testResults.Count; i++)
            {
                double max = -1;
                int maxIndex = -1;
                for (int j = 0; j < result[i].Length; j++)
                {
                    if (result[i][j] > max)
                    {
                        max = result[i][j];
                        maxIndex = j;
                    }
                }
                Console.WriteLine("Expected: {0} Actual: {1} Output {2}", testResults[i], maxIndex + 1, max);
            }
        }

        private static List<IEnumerable<double>> ReadData(string fileName)
        {
            var data = new List<IEnumerable<double>>();

            using (var sr = new StreamReader(fileName))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    var dataEntry = line.Split(',').Select(x => double.Parse(x, CultureInfo.InvariantCulture));
                    data.Add(dataEntry);
                }
            }
            return data;
        }
    }
}
