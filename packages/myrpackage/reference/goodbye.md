# Say Goodbye

This function prints a friendly farewell. It can be customized with
different names and languages.

## Usage

``` r
goodbye(name = "world", language = "english", exclamation = TRUE, capitalize = FALSE)
```

## Arguments

- name:

  Character string. The name to bid farewell to. Default is "world".

- language:

  Character string. The language for the farewell. Supported languages:
  "english" (default), "spanish", "french", "portuguese", "german",
  "italian".

- exclamation:

  Logical. Whether to add an exclamation mark. Default is TRUE.

- capitalize:

  Logical. Whether to capitalize the first letter of the name. Default
  is FALSE.

## Value

A character string containing the farewell message.

## See also

[`hello`](https://diogoribeiro7.github.io/packages/myrpackage/reference/hello.md)
for a greeting function

## Examples

``` r
goodbye()
#> Goodbye, world! 
goodbye("R Users")
#> Goodbye, R Users! 
goodbye("amigos", language = "spanish")
#> Adiós, amigos! 
goodbye("mes amis", language = "french", exclamation = FALSE)
#> Au revoir, mes amis. 
goodbye("world", capitalize = TRUE)
#> Goodbye, World! 
```
