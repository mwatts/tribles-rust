use proc_macro2::Span;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Token};

/// A single variable declaration inside `find!((var: Type?, ...), constraint)`.
struct FindVariable {
    name: Ident,
    ty: Option<syn::Type>,
    /// When true the variable yields `Result<T, E>` and does not filter.
    fallible: bool,
}

/// Parsed input for `__find_impl!(crate_path, ctx, (vars...), constraint)`.
struct FindImplInput {
    crate_path: syn::Path,
    _comma1: Token![,],
    ctx: Ident,
    _comma2: Token![,],
    variables: Vec<FindVariable>,
    _comma3: Token![,],
    constraint: TokenStream2,
}

impl Parse for FindImplInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let crate_path: syn::Path = input.parse()?;
        let _comma1: Token![,] = input.parse()?;
        let ctx: Ident = input.parse()?;
        let _comma2: Token![,] = input.parse()?;

        // Parse the parenthesised variable list.
        let vars_content;
        syn::parenthesized!(vars_content in input);
        let mut variables = Vec::new();
        while !vars_content.is_empty() {
            let name: Ident = vars_content.parse()?;

            // Optional `: Type`
            let ty = if vars_content.peek(Token![:]) {
                vars_content.parse::<Token![:]>()?;
                Some(vars_content.parse::<syn::Type>()?)
            } else {
                None
            };

            // Optional trailing `?`
            let fallible = if vars_content.peek(Token![?]) {
                vars_content.parse::<Token![?]>()?;
                true
            } else {
                false
            };

            variables.push(FindVariable { name, ty, fallible });

            // Consume a trailing comma if present.
            if vars_content.peek(Token![,]) {
                vars_content.parse::<Token![,]>()?;
            }
        }

        let _comma3: Token![,] = input.parse()?;
        let constraint: TokenStream2 = input.parse()?;

        Ok(FindImplInput {
            crate_path,
            _comma1,
            ctx,
            _comma2,
            variables,
            _comma3,
            constraint,
        })
    }
}

pub fn find_impl(input: TokenStream2) -> syn::Result<TokenStream2> {
    let FindImplInput {
        crate_path,
        ctx,
        variables,
        constraint,
        ..
    } = syn::parse2(input)?;

    let binding = format_ident!("__binding", span = Span::mixed_site());

    // Generate `let var = ctx.next_variable();` for each variable.
    let var_decls: Vec<TokenStream2> = variables
        .iter()
        .map(|v| {
            let name = &v.name;
            quote! { let #name = #ctx.next_variable(); }
        })
        .collect();

    // Generate conversion code inside the closure for each variable.
    let var_conversions: Vec<TokenStream2> = variables
        .iter()
        .map(|v| {
            let name = &v.name;
            if v.fallible {
                // `?` variable: yield Result<T, E>, no filtering.
                if let Some(ref ty) = v.ty {
                    quote! {
                        let #name: ::core::result::Result<#ty, _> =
                            #crate_path::value::TryFromValue::try_from_value(#name.extract(#binding));
                    }
                } else {
                    quote! {
                        let #name =
                            #crate_path::value::TryFromValue::try_from_value(#name.extract(#binding));
                    }
                }
            } else {
                // Default: filter on conversion failure.
                if let Some(ref ty) = v.ty {
                    quote! {
                        let #name: #ty = match #crate_path::value::TryFromValue::try_from_value(
                            #name.extract(#binding)
                        ) {
                            ::core::result::Result::Ok(__v) => __v,
                            ::core::result::Result::Err(_) => return ::core::option::Option::None,
                        };
                    }
                } else {
                    quote! {
                        let #name = match #crate_path::value::TryFromValue::try_from_value(
                            #name.extract(#binding)
                        ) {
                            ::core::result::Result::Ok(__v) => __v,
                            ::core::result::Result::Err(_) => return ::core::option::Option::None,
                        };
                    }
                }
            }
        })
        .collect();

    // Build the result tuple.
    let var_names: Vec<&Ident> = variables.iter().map(|v| &v.name).collect();
    let tuple_expr = match var_names.len() {
        0 => quote! { () },
        1 => {
            let v = var_names[0];
            quote! { (#v,) }
        }
        _ => {
            quote! { (#(#var_names),*) }
        }
    };

    let output = quote! {
        {
            #(#var_decls)*
            #crate_path::query::Query::new(#constraint,
                move |#binding| {
                    #(#var_conversions)*
                    ::core::option::Option::Some(#tuple_expr)
                }
            )
        }
    };

    Ok(output)
}
