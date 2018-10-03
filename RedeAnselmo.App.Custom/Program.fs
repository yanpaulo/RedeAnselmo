namespace RedeAnselmo

// Saiba mais sobre F# em http://fsharp.org
// Veja o projeto 'F# Tutorial' para obter mais ajuda.
module Application =
    [<EntryPoint>]
    let main argv = 
        printfn "%A" (AlgoritmoClassificacao.iris ())
        0 // retornar um código de saída inteiro
