namespace RedeAnselmo

open System
open System.Diagnostics

open FSharp.Data
open MathNet.Numerics
open MathNet.Numerics.Random
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics
open FSharp.Collections.ParallelSeq

module Algoritmo =
    //Tipos
    
    //Par de Entrada X e saída desejada Y
    type Par = { X: float Vector; Y: float Vector }
    //Modelo contendo as listas de neurônios I (camada oculta) e J (camada de saída)
    type Modelo = { I: float Matrix; J: float Matrix }
    //Parâmetros de entrada para a realização do algoritmo
    type Entrada = { Dados: Par list; Classes: Vector<float> list; NumeroNeuronios: float }
    
    let rng = Random.shared
    let pow x n = Math.Pow(x, n)
    let pow2 x = x * x
    let e = Math.E

    //Funções
    let radial x u =
        (x: float Vector) |> ignore
        (u: float Vector) |> ignore
        let termo = Distance.Euclidean(x, u) |> pow2
        
        pow e -termo
    
    let sigmoide x = 
        1.0 /  (1.0 + Math.Pow(Math.E, -x))

    let transferencia x = sigmoide x

    let saidaI i x =
        let s = x * i |> Seq.map (fun n -> n |> transferencia) |> List.ofSeq
        1.0 :: s |> vector

    //Resultado para entrada x na rede
    let resultado m x =
        let x = saidaI m.I x
        let res = m.J * x
        res.Map(fun v -> Math.Round(v))
        
    
    //Normalização
    let normaliza x min max =
        (x: float) |> ignore
        (min: float) |> ignore
        (max: float) |> ignore

        (x - min) / (max - min)
    
    //Contador de tempo para medição de performance.
    let sw = new Stopwatch();
    
    //Próximo modelo para o vetor "dados"
    //Implementação back-propagation com tail-recursion
    let pesos dados numNeuronios  =
        (dados: Par list) |> ignore
        
        let I = Matrix<float>.Build.Random(dados.Head.X.Count, numNeuronios)

        let X = dados |> List.map (fun p -> saidaI I p.X) |> Matrix.Build.DenseOfColumnVectors
        let Y = dados |> List.map (fun p -> p.Y) |> Matrix.Build.DenseOfColumnVectors
            
            //let W = Y * ((X.Transpose() * X).Inverse() * X.Transpose())
        let W = Y * X.PseudoInverse()

        { I = I; J = W }
     

