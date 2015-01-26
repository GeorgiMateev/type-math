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
    public class MnistData
    {
        public void Learn()
        {
            var trainDataFileName = "train-images.idx3-ubyte";
            var trainDataLabels = "train-labels.idx1-ubyte";
            var data = MnistData.ReadImageData(trainDataFileName);
            var labels = MnistData.ReadLabelsData(trainDataLabels);

            var hiddenLayerNeurons = 200;
            var iterations = 500;
            var learningConst = 0.1;
            var net = new Network(784, 10, hiddenLayerNeurons);
            net.Train(data, labels, iterations, learningConst);

            var testFileName = "t10k-images.idx3-ubyte";
            var resultsFileName = "t10k-labels.idx1-ubyte";

            var testCases = MnistData.ReadImageData(testFileName);
            var testResults = MnistData.ReadLabelsData(resultsFileName);

            var result = net.ComputeData(testCases);

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

        private static List<IEnumerable<double>> ReadImageData(string fileName)
        {
            var data = new List<IEnumerable<double>>();

            using (var fs = new FileStream(fileName, FileMode.Open))
            {
                var magicNumber = ReadLine(fs);
                var numberOfImages = ReadLine(fs);
                var numberOfRows = ReadLine(fs);
                var numberOfColumns = ReadLine(fs);

                for (int i = 0; i < numberOfImages; i++)
                {
                    var pixels = new List<double>();
                    var numberOfPixels = numberOfRows * numberOfColumns;

                    for (int j = 0; j < numberOfPixels; j++)
                    {
                        var pixel = fs.ReadByte();
                        var inputData = pixel / 255.0d;
                        pixels.Add(inputData);
                    }

                    data.Add(pixels);
                }
            }

            return data;
        }

        private static List<int> ReadLabelsData(string fileName)
        {
            var labels = new List<int>();

            using (var fs = new FileStream(fileName, FileMode.Open))
            {
                var magicNumber = ReadLine(fs);
                var numberOfItems = ReadLine(fs);

                for (int i = 0; i < numberOfItems; i++)
                {
                    var label = fs.ReadByte();
                    labels.Add(label);
                }
            }

            return labels;
        }

        private static int ReadLine(FileStream fs)
        {
            return (fs.ReadByte() << 24) | (fs.ReadByte() << 16) | (fs.ReadByte() << 8) | fs.ReadByte();
        }
    }
}
