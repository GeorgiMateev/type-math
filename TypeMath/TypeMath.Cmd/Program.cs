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
    class Program
    {
        static void Main(string[] args)
        {
            //var iris = new IrisData();
            //iris.Learn();

            var mnist = new MnistData();
            mnist.Learn();
        }        
    }
}
