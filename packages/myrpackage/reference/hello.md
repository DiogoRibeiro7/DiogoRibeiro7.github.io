# Say Hello

This function prints a friendly greeting. It can be customized with
different names and languages.

## Usage

``` r
hello(name = "world", language = "english", exclamation = TRUE, capitalize = FALSE)
```

## Arguments

- name:

  Character string. The name to greet. Default is "world".

- language:

  Character string. The language for the greeting. Supported languages:
  "english" (default), "spanish", "french", "portuguese", "german",
  "italian".

- exclamation:

  Logical. Whether to add an exclamation mark. Default is TRUE.

- capitalize:

  Logical. Whether to capitalize the first letter of the name. Default
  is FALSE.

## Value

A character string containing the greeting.

## See also

[`goodbye`](https://diogoribeiro7.github.io/packages/myrpackage/reference/goodbye.md)
for a farewell function

## Examples

``` r
hello()
#> Hello, world! 
hello("R Users")
#> Hello, R Users! 
hello("amigos", language = "spanish")
#> Hola, amigos! 
hello("mes amis", language = "french", exclamation = FALSE)
#> Bonjour, mes amis. 
hello("world", capitalize = TRUE)
#> Hello, World! 
```
