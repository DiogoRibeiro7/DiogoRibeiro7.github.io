# Advanced Usage of myrpackage

``` r
library(myrpackage)
```

## Advanced Usage Patterns

This vignette covers more advanced usage patterns for the `myrpackage`
functions.

### Creating Custom Greeting Functions

You can create custom greeting functions based on
[`hello()`](https://diogoribeiro7.github.io/packages/myrpackage/reference/hello.md)
and
[`goodbye()`](https://diogoribeiro7.github.io/packages/myrpackage/reference/goodbye.md)
to maintain consistent greetings throughout your application:

``` r
# Create a custom Spanish greeter
spanish_hello <- function(name = "amigos", exclamation = TRUE, capitalize = FALSE) {
  hello(name = name, language = "spanish", exclamation = exclamation, capitalize = capitalize)
}

# Try it out
spanish_hello("todos")
#> Hola, todos!
spanish_hello("R usuarios", capitalize = TRUE)
#> Hola, R usuarios!
```

### Working with Multiple Languages

If your application needs to support multiple languages, you can create
a wrapper function:

``` r
multilingual_greeting <- function(name, languages = c("english", "spanish", "french")) {
  for (lang in languages) {
    hello(name = name, language = lang)
  }
}

# Greet in multiple languages
multilingual_greeting("R users", c("english", "french", "german"))
#> Hello, R users! 
#> Bonjour, R users! 
#> Hallo, R users!
```

### Creating Personalized Messages

You can combine the greeting functions with other text to create more
personalized messages:

``` r
create_welcome_message <- function(name, language = "english") {
  greeting <- suppressMessages(hello(name, language, exclamation = TRUE))

  # Add a personalized message
  additional_text <- switch(
    language,
    english = "Welcome to our application.",
    spanish = "Bienvenido a nuestra aplicación.",
    french = "Bienvenue dans notre application.",
    portuguese = "Bem-vindo ao nosso aplicativo.",
    german = "Willkommen in unserer Anwendung.",
    italian = "Benvenuto nella nostra applicazione.",
    "Welcome to our application."
  )

  message <- paste(greeting, additional_text)
  cat(message, "\n")

  invisible(message)
}

# Create a welcome message
create_welcome_message("John", "english")
#> Hello, John! 
#> Hello, John! Welcome to our application.
create_welcome_message("Maria", "spanish")
#> Hola, Maria! 
#> Hola, Maria! Bienvenido a nuestra aplicación.
```

### Error Handling

When working with user input, it’s good practice to handle potential
errors:

``` r
safe_hello <- function(name, language) {
  tryCatch(
    {
      hello(name, language)
    },
    error = function(e) {
      cat("Error in greeting:", conditionMessage(e), "\n")
      cat("Using default greeting instead.\n")
      hello()
    },
    warning = function(w) {
      cat("Warning:", conditionMessage(w), "\n")
      # Continue execution with the warning
      suppressWarnings(hello(name, language))
    }
  )
}

# Test with valid inputs
safe_hello("R users", "english")
#> Hello, R users!

# Test with an invalid language (produces a warning)
safe_hello("R users", "klingon")
#> Warning: Language 'klingon' not supported. Using English instead. Supported languages: english, spanish, french, portuguese, german, italian 
#> Hello, R users!

# Test with an invalid name (produces an error)
safe_hello(c("multiple", "names"), "english")
#> Error in greeting: 'name' must be a single character string 
#> Using default greeting instead.
#> Hello, world!
```

### Integration with Other Packages

You can easily integrate `myrpackage` functions with other packages,
such as creating a shiny app:

``` r
# This is an example, not run
library(shiny)

ui <- fluidPage(
  titlePanel("Multilingual Greeter"),

  sidebarLayout(
    sidebarPanel(
      textInput("name", "Your Name:", "world"),
      selectInput("language", "Language:",
                  choices = c("English" = "english",
                             "Spanish" = "spanish",
                             "French" = "french",
                             "Portuguese" = "portuguese",
                             "German" = "german",
                             "Italian" = "italian")),
      checkboxInput("exclamation", "Add Exclamation Mark", TRUE),
      checkboxInput("capitalize", "Capitalize Name", FALSE)
    ),

    mainPanel(
      h3("Greeting:"),
      verbatimTextOutput("greeting"),
      h3("Farewell:"),
      verbatimTextOutput("farewell")
    )
  )
)

server <- function(input, output) {
  output$greeting <- renderText({
    suppressMessages(
      hello(input$name, input$language, input$exclamation, input$capitalize)
    )
  })

  output$farewell <- renderText({
    suppressMessages(
      goodbye(input$name, input$language, input$exclamation, input$capitalize)
    )
  })
}

# shinyApp(ui = ui, server = server)
```

### Performance Considerations

For high-volume applications, you might want to avoid printing to the
console:

``` r
# Create a silent version of hello() that doesn't print to console
silent_hello <- function(name = "world", language = "english", exclamation = TRUE, capitalize = FALSE) {
  # Input validation - simplified for brevity
  if (!is.character(name) || length(name) != 1) {
    stop("'name' must be a single character string")
  }

  # Capitalize name if requested
  if (capitalize) {
    name <- paste0(toupper(substr(name, 1, 1)), substr(name, 2, nchar(name)))
  }

  # Select greeting based on language
  greeting <- switch(
    language,
    english = "Hello",
    spanish = "Hola",
    french = "Bonjour",
    portuguese = "Olá",
    german = "Hallo",
    italian = "Ciao",
    "Hello"
  )

  # Construct the greeting
  result <- paste0(greeting, ", ", name)

  # Add exclamation mark if requested
  if (exclamation) {
    result <- paste0(result, "!")
  } else {
    result <- paste0(result, ".")
  }

  # Return the greeting (without printing)
  return(result)
}

# Use the silent version
greeting <- silent_hello("R users", "french")
greeting
#> [1] "Bonjour, R users!"
```

### Batch Processing

You can use the greeting functions in batch processing scenarios:

``` r
# Create greetings for a list of names
names <- c("Alice", "Bob", "Charlie", "David")
languages <- c("english", "spanish", "french", "portuguese")

# Create a dataframe with all combinations
greetings_df <- expand.grid(name = names, language = languages, stringsAsFactors = FALSE)

# Add greetings
greetings_df$greeting <- mapply(
  function(name, language) {
    suppressMessages(hello(name, language))
  },
  greetings_df$name,
  greetings_df$language
)
#> Hello, Alice! 
#> Hello, Bob! 
#> Hello, Charlie! 
#> Hello, David! 
#> Hola, Alice! 
#> Hola, Bob! 
#> Hola, Charlie! 
#> Hola, David! 
#> Bonjour, Alice! 
#> Bonjour, Bob! 
#> Bonjour, Charlie! 
#> Bonjour, David! 
#> Olá, Alice! 
#> Olá, Bob! 
#> Olá, Charlie! 
#> Olá, David!

# Display the first few rows
head(greetings_df)
#>      name language        greeting
#> 1   Alice  english   Hello, Alice!
#> 2     Bob  english     Hello, Bob!
#> 3 Charlie  english Hello, Charlie!
#> 4   David  english   Hello, David!
#> 5   Alice  spanish    Hola, Alice!
#> 6     Bob  spanish      Hola, Bob!
```

## Conclusion

These examples demonstrate how the simple greeting functions in
`myrpackage` can be extended and integrated into more complex
applications. By following R package best practices, even simple
functions can be made robust, flexible, and user-friendly.
