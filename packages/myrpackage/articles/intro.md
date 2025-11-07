# Getting Started with myrpackage

``` r
library(myrpackage)
```

## Introduction

The `myrpackage` package provides simple greeting functions that
demonstrate proper R package structure and documentation. This vignette
introduces the main functionality of the package and shows examples of
how to use it.

## Basic Usage

The package contains two main functions:

- [`hello()`](https://diogoribeiro7.github.io/packages/myrpackage/reference/hello.md):
  Say hello in various languages
- [`goodbye()`](https://diogoribeiro7.github.io/packages/myrpackage/reference/goodbye.md):
  Say goodbye in various languages

### Saying Hello

The
[`hello()`](https://diogoribeiro7.github.io/packages/myrpackage/reference/hello.md)
function generates a friendly greeting. By default, it says “Hello,
world!”:

``` r
hello()
#> Hello, world!
```

You can customize the greeting with a different name:

``` r
hello("R Users")
#> Hello, R Users!
```

### Multilingual Greetings

Both functions support multiple languages. Currently supported languages
are:

- English (default)
- Spanish
- French
- Portuguese
- German
- Italian

Here are examples of greetings in different languages:

``` r
hello("amigos", language = "spanish")
#> Hola, amigos!
hello("mes amis", language = "french")
#> Bonjour, mes amis!
hello("amigos", language = "portuguese")
#> Olá, amigos!
hello("freunde", language = "german")
#> Hallo, freunde!
hello("amici", language = "italian")
#> Ciao, amici!
```

### Customizing Punctuation

You can also control whether the greeting ends with an exclamation mark:

``` r
hello(exclamation = TRUE)  # Default
#> Hello, world!
hello(exclamation = FALSE)
#> Hello, world.
```

### Capitalizing Names

You can capitalize the first letter of the name:

``` r
hello("r users", capitalize = TRUE)
#> Hello, R users!
```

### Saying Goodbye

The
[`goodbye()`](https://diogoribeiro7.github.io/packages/myrpackage/reference/goodbye.md)
function works similarly to
[`hello()`](https://diogoribeiro7.github.io/packages/myrpackage/reference/hello.md),
but provides farewell messages:

``` r
goodbye()
#> Goodbye, world!
goodbye("R Users")
#> Goodbye, R Users!
goodbye("amigos", language = "spanish")
#> Adiós, amigos!
goodbye("mes amis", language = "french", exclamation = FALSE)
#> Au revoir, mes amis.
```

## Error Handling

Both functions include input validation and helpful error messages:

``` r
# These will generate errors:
hello(name = c("world", "everyone"))
#> Error in hello(name = c("world", "everyone")): 'name' must be a single character string
```

``` r
hello(exclamation = "yes")
#> Error in hello(exclamation = "yes"): 'exclamation' must be a single logical value
```

The functions also provide helpful warnings when unsupported languages
are requested:

``` r
hello("world", language = "klingon")
#> Warning in hello("world", language = "klingon"): Language 'klingon' not
#> supported. Using English instead. Supported languages: english, spanish,
#> french, portuguese, german, italian
#> Hello, world!
```

## Further Reading

For complete details about function arguments and behavior, refer to the
function documentation:

``` r
?hello
?goodbye
```

For more advanced usage patterns, see the “Advanced Usage” vignette:

``` r
vignette("advanced", package = "myrpackage")
```

## Conclusion

`myrpackage` is a simple demonstration package that showcases proper R
package development practices. It includes:

- Well-documented functions
- Comprehensive tests
- Properly structured package components
- Vignettes for user documentation
- Input validation and error handling

While the functionality is minimal, the package structure follows best
practices for R package development.
