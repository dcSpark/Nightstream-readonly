use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, spanned::Spanned, Data, DeriveInput, Fields, ItemFn, ReturnType, Type};

fn reject_attr_args(attr: TokenStream, name: &str) -> Result<(), TokenStream> {
    if attr.is_empty() {
        return Ok(());
    }
    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        format!("`#[nightstream_sdk::{name}]` does not take arguments"),
    )
    .to_compile_error()
    .into())
}

fn is_never_type(ty: &Type) -> bool {
    matches!(ty, Type::Never(_))
}

#[proc_macro_derive(NeoAbi)]
pub fn derive_neo_abi(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let ident = &input.ident;
    if !input.generics.params.is_empty() {
        return syn::Error::new(
            input.generics.span(),
            "`#[derive(NeoAbi)]` does not support generics yet",
        )
        .to_compile_error()
        .into();
    }

    let Data::Struct(data_struct) = &input.data else {
        return syn::Error::new(input.span(), "`#[derive(NeoAbi)]` only supports structs")
            .to_compile_error()
            .into();
    };

    let (words_expr, read_body, write_body) = match &data_struct.fields {
        Fields::Named(fields) => {
            let field_idents: Vec<_> = fields
                .named
                .iter()
                .map(|f| f.ident.clone().expect("named field ident"))
                .collect();
            let field_tys: Vec<_> = fields.named.iter().map(|f| f.ty.clone()).collect();

            let words = quote! { 0 #( + <#field_tys as ::nightstream_sdk::abi::NeoAbi>::WORDS )* };

            let reads = field_idents.iter().zip(field_tys.iter()).map(|(name, ty)| {
                quote! {
                    let #name: #ty = <#ty as ::nightstream_sdk::abi::NeoAbi>::read_from_words(p);
                    p = p.add(<#ty as ::nightstream_sdk::abi::NeoAbi>::WORDS);
                }
            });

            let writes = field_idents.iter().zip(field_tys.iter()).map(|(name, ty)| {
                quote! {
                    <#ty as ::nightstream_sdk::abi::NeoAbi>::write_to_words(&self.#name, p);
                    p = p.add(<#ty as ::nightstream_sdk::abi::NeoAbi>::WORDS);
                }
            });

            (
                words,
                quote! {
                    let mut p = ptr;
                    #(#reads)*
                    Self { #(#field_idents),* }
                },
                quote! {
                    let mut p = ptr;
                    #(#writes)*
                },
            )
        }
        Fields::Unnamed(fields) => {
            let field_tys: Vec<_> = fields.unnamed.iter().map(|f| f.ty.clone()).collect();
            let field_idxs: Vec<_> = (0..field_tys.len()).map(syn::Index::from).collect();
            let locals: Vec<_> = (0..field_tys.len())
                .map(|i| syn::Ident::new(&format!("f{i}"), proc_macro2::Span::call_site()))
                .collect();

            let words = quote! { 0 #( + <#field_tys as ::nightstream_sdk::abi::NeoAbi>::WORDS )* };

            let reads = locals.iter().zip(field_tys.iter()).map(|(name, ty)| {
                quote! {
                    let #name: #ty = <#ty as ::nightstream_sdk::abi::NeoAbi>::read_from_words(p);
                    p = p.add(<#ty as ::nightstream_sdk::abi::NeoAbi>::WORDS);
                }
            });

            let writes = field_idxs.iter().zip(field_tys.iter()).map(|(idx, ty)| {
                quote! {
                    <#ty as ::nightstream_sdk::abi::NeoAbi>::write_to_words(&self.#idx, p);
                    p = p.add(<#ty as ::nightstream_sdk::abi::NeoAbi>::WORDS);
                }
            });

            (
                words,
                quote! {
                    let mut p = ptr;
                    #(#reads)*
                    Self(#(#locals),*)
                },
                quote! {
                    let mut p = ptr;
                    #(#writes)*
                },
            )
        }
        Fields::Unit => (
            quote! { 0usize },
            quote! { Self },
            quote! {},
        ),
    };

    let expanded = quote! {
        impl ::nightstream_sdk::abi::NeoAbi for #ident {
            const WORDS: usize = #words_expr;

            #[inline]
            unsafe fn read_from_words(ptr: *const u32) -> Self {
                #read_body
            }

            #[inline]
            unsafe fn write_to_words(&self, ptr: *mut u32) {
                #write_body
            }
        }
    };

    expanded.into()
}

#[proc_macro_attribute]
pub fn entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Err(ts) = reject_attr_args(attr, "entry") {
        return ts;
    }

    let f = parse_macro_input!(item as ItemFn);
    if !f.sig.inputs.is_empty() {
        return syn::Error::new(
            f.sig.inputs.span(),
            "`#[nightstream_sdk::entry]` function must take no arguments",
        )
            .to_compile_error()
            .into();
    }

    let call_ident = &f.sig.ident;
    let wrapper_ident = syn::Ident::new(
        &format!("__nightstream_sdk_entry_{}", call_ident),
        call_ident.span(),
    );
    let returns_never = matches!(
        &f.sig.output,
        ReturnType::Type(_, ty) if is_never_type(&**ty)
    );

    let asm = format!(
        r#"
    .section .neo_start,"ax",@progbits
    .globl _start
    .extern _STACK_PTR
_start:
    la sp, _STACK_PTR
    call {wrapper}
    ecall
1:
    j 1b
"#,
        wrapper = wrapper_ident
    );

    let wrapper_body = if returns_never {
        quote! { #call_ident(); }
    } else {
        quote! {
            let _ = #call_ident();
            ::nightstream_sdk::halt()
        }
    };

    let expanded = quote! {
        #f

        #[no_mangle]
        extern "C" fn #wrapper_ident() -> ! {
            #wrapper_body
        }

        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        ::core::arch::global_asm!(#asm);

        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
        compile_error!("`#[nightstream_sdk::entry]` is only supported on RISC-V targets");
    };

    expanded.into()
}

#[proc_macro_attribute]
pub fn provable(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Err(ts) = reject_attr_args(attr, "provable") {
        return ts;
    }

    let f = parse_macro_input!(item as ItemFn);

    let mut arg_idents: Vec<syn::Ident> = Vec::new();
    let mut arg_types: Vec<Type> = Vec::new();
    for arg in &f.sig.inputs {
        let syn::FnArg::Typed(pat_ty) = arg else {
            return syn::Error::new(
                arg.span(),
                "`#[nightstream_sdk::provable]` does not support methods/self arguments",
            )
            .to_compile_error()
            .into();
        };
        let ty = (*pat_ty.ty).clone();
        let syn::Pat::Ident(pat_ident) = &*pat_ty.pat else {
            return syn::Error::new(
                pat_ty.pat.span(),
                "`#[nightstream_sdk::provable]` arguments must be simple identifiers (e.g. `n: u32`)",
            )
            .to_compile_error()
            .into();
        };
        arg_idents.push(pat_ident.ident.clone());
        arg_types.push(ty);
    }

    let call_ident = &f.sig.ident;
    let wrapper_ident = syn::Ident::new(
        &format!("__nightstream_sdk_provable_{}", call_ident),
        call_ident.span(),
    );

    let (ret_ty, returns_never) = match &f.sig.output {
        ReturnType::Type(_, ty) => ((**ty).clone(), is_never_type(&**ty)),
        ReturnType::Default => (syn::parse_quote! { () }, false),
    };

    let asm = format!(
        r#"
    .section .neo_start,"ax",@progbits
    .globl _start
    .extern _STACK_PTR
_start:
    la sp, _STACK_PTR
    call {wrapper}
    ecall
1:
    j 1b
"#,
        wrapper = wrapper_ident
    );

    let arg_stmts = arg_idents
        .iter()
        .zip(arg_types.iter())
        .map(|(ident, ty)| {
            quote! {
                let #ident: #ty = unsafe {
                    let v = <#ty as ::nightstream_sdk::abi::NeoAbi>::read_from_words(__neo_ptr);
                    __neo_ptr = __neo_ptr.add(<#ty as ::nightstream_sdk::abi::NeoAbi>::WORDS);
                    v
                };
            }
        });

    let wrapper_body = if returns_never {
        quote! {
            let mut __neo_ptr = ::nightstream_sdk::input_addr() as *const u32;
            #(#arg_stmts)*
            #call_ident(#(#arg_idents),*);
        }
    } else {
        quote! {
            let mut __neo_ptr = ::nightstream_sdk::input_addr() as *const u32;
            #(#arg_stmts)*
            let out: #ret_ty = #call_ident(#(#arg_idents),*);
            let __neo_out = ::nightstream_sdk::output_addr() as *mut u32;
            unsafe { <#ret_ty as ::nightstream_sdk::abi::NeoAbi>::write_to_words(&out, __neo_out) };
            ::nightstream_sdk::halt()
        }
    };

    let expanded = quote! {
        #f

        #[no_mangle]
        extern "C" fn #wrapper_ident() -> ! {
            #wrapper_body
        }

        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        ::core::arch::global_asm!(#asm);

        #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
        compile_error!("`#[nightstream_sdk::provable]` is only supported on RISC-V targets");
    };

    expanded.into()
}
