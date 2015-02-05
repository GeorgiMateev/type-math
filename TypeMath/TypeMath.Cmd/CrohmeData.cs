using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TypeMath.NeuralNetwork;

namespace TypeMath.Cmd
{
    class CrohmeData
    {
        public async void Learn()
        {
            var startTime = DateTime.Now;

            var trainDataFolder = "../../../data/crohme/train";
            var testDataFolder = "../../../data/crohme/test";

            int? lastUsedClass = -1;
            var labelRepresentations = new Dictionary<string, int>();

            List<IEnumerable<double>> trainData;
            List<int> trainLabels;
            GetData(trainDataFolder, out trainData, out trainLabels,  labelRepresentations, ref lastUsedClass);
            
            var hiddenLayerNeurons = 10;
            var iterations = 5;
            var learningConst = 0.1;

            var inputNeurons = trainData[0].Count();
            var outputNeurons = lastUsedClass.Value + 1;
            var net = new Network(inputNeurons, outputNeurons, hiddenLayerNeurons);
            net.Train(trainData, trainLabels, iterations, learningConst);

            List<IEnumerable<double>> testData;
            List<int> testLabels;
            int? lastClass = null;
            GetData(testDataFolder, out testData, out testLabels, labelRepresentations, ref lastClass);

            var networkResults = net.Classify(testData);

            var endTime = DateTime.Now;

            double successfullTests = 0;

            for (int i = 0; i < testLabels.Count; i++)
            {
                double max = -1;
                int maxIndex = -1;
                for (int j = 0; j < networkResults[i].Length; j++)
                {
                    if (networkResults[i][j] > max)
                    {
                        max = networkResults[i][j];
                        maxIndex = j;
                    }
                }

                var testLabel = GetLabel(testLabels[i], labelRepresentations);
                var recognizedLabel = GetLabel(maxIndex, labelRepresentations);
                Console.WriteLine("Expected: {0} Actual: {1} Output {2}", testLabel, recognizedLabel, max);

                if (testLabel == recognizedLabel)
                {
                    successfullTests++;
                }
            }

            var time = endTime - startTime;

            Console.WriteLine("Execution time: {0}m {1}s", time.Minutes, time.Seconds);

            Console.WriteLine("Training samples: {0}", trainData.Count);
            Console.WriteLine("Successfull tests: {0}", successfullTests);
            Console.WriteLine("Success rate: {0}", successfullTests / testData.Count);
        }

        private static void GetData(
            string dataFolder,
            out List<IEnumerable<double>> data,
            out List<int> labels,
            Dictionary<string, int> labelRepresentations,
            ref int? lastUsedClass)
        {
            data = new List<IEnumerable<double>>();
            labels = new List<int>();

            var trainFiles = Directory.GetFiles(dataFolder);

            foreach (var file in trainFiles)
            {
                string label;
                var trainSymbol = CrohmeData.GetTrainSymbol(file, out label);
                var trainClass = CrohmeData.GetClass(label, labelRepresentations, ref lastUsedClass);

                data.Add(trainSymbol);
                labels.Add(trainClass);
            }
        }

        private static int GetClass(string label, Dictionary<string, int> labelRepresentations, ref int? lastUsedClass)
        {
            int labelClass;
            if (labelRepresentations.TryGetValue(label, out labelClass))
            {
                return labelClass;
            }
            else
            {
                if (!lastUsedClass.HasValue)
                {
                    throw new ArgumentException("There is no such label in the label representations.");
                }

                labelClass = lastUsedClass.Value + 1;
                lastUsedClass++;
                labelRepresentations.Add(label, lastUsedClass.Value);
                return labelClass;
            }
        }

        private static string GetLabel(int labelClass, Dictionary<string, int> labelRepresentations)
        {
            return labelRepresentations.FirstOrDefault(kv => kv.Value == labelClass).Key;
        }

        private static IEnumerable<double> GetTrainSymbol(string file, out string label)
        {
            using (var sr = new StreamReader(file))
            {
                label = sr.ReadLine();
                var content = sr.ReadToEnd();
                var attributes = content.Split(
                    new string[] { Environment.NewLine, " " },
                    StringSplitOptions.RemoveEmptyEntries)
                    .Select(c => double.Parse(c));

                return attributes;
            }
        }
    }
}
