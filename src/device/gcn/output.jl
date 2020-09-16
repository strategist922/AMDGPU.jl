export OutputContext, @rocprint, @rocprintln, @rocprintf, @rocprintfw

"Internal representation of a static string."
struct DeviceStaticString{N} end
Base.sizeof(dss::DeviceStaticString{N}) where N = N

function Base.unsafe_load(ptr::DevicePtr{DeviceStaticString{N},AS.Global}) where N
    vec_ptr = convert(Ptr{UInt8}, ptr)
    vec_raw = Base.unsafe_wrap(Vector{UInt8}, vec_ptr, (N,))
    idx = findfirst(x->x==0, vec_raw)
    idx = idx === nothing ? N : idx
    return vec_raw[1:idx-1]
end
Base.unsafe_store!(ptr::DevicePtr{<:DeviceStaticString}, x) = nothing

struct OutputContext{HC}
    hostcall::HC
end
function OutputContext(io::IO=stdout; agent=get_default_agent(), buf_len=2^16, kwargs...)
    hc = HostCall(Int64, Tuple{DeviceStaticString{buf_len}}; agent=agent, continuous=true, kwargs...) do bytes
        print(io, String(bytes))
        Int64(length(bytes))
    end
    OutputContext(hc)
end

const GLOBAL_OUTPUT_CONTEXT_TYPE = OutputContext{HostCall{UInt64,Int64,Tuple{DeviceStaticString{2^16}}}}

### macros

macro rocprint(oc, str)
    rocprint(oc, str)
end
macro rocprintln(oc, str)
    rocprint(oc, str, true)
end

macro rocprint(str)
    @gensym oc_ptr oc
    ex = quote
        $(esc(oc_ptr)) = AMDGPU.get_global_pointer(Val(:__global_output_context),
                                                         $GLOBAL_OUTPUT_CONTEXT_TYPE)
        $(esc(oc)) = Base.unsafe_load($(esc(oc_ptr)))
    end
    push!(ex.args, rocprint(oc, str))
    ex
end
macro rocprintln(str)
    @gensym oc_ptr oc
    ex = quote
        $(esc(oc_ptr)) = AMDGPU.get_global_pointer(Val(:__global_output_context),
                                                         $GLOBAL_OUTPUT_CONTEXT_TYPE)
        $(esc(oc)) = Base.unsafe_load($(esc(oc_ptr)))
    end
    push!(ex.args, rocprint(oc, str, true))
    ex
end

### parse-time helpers

function rocprint(oc, str, nl::Bool=false)
    ex = Expr(:block)
    if !(str isa Expr)
        str = Expr(:string, str)
    end
    @assert str.head == :string
    for (idx,arg) in enumerate(str.args)
        if nl && idx == length(str.args)
            arg *= '\n'
        end
        N = rocprint!(ex, 1, oc, arg)
        N = rocprint!(ex, N, oc, '\0')
        dstr = DeviceStaticString{N}()
        push!(ex.args, :(hostcall!($(esc(oc)).hostcall, $dstr)))
    end
    push!(ex.args, :(nothing))
    return ex
end
function rocprint!(ex, N, oc, str::String)
    @gensym str_ptr
    push!(ex.args, :($str_ptr = AMDGPU.alloc_string($(Val(Symbol(str))))))
    push!(ex.args, :(AMDGPU.memcpy!($(esc(oc)).hostcall.buf_ptr+$(N-1), $str_ptr, $(length(str)))))
    return N+length(str)
end
function rocprint!(ex, N, oc, char::Char)
    @assert char == '\0' "Non-null chars not yet implemented"
    byte = UInt8(char)
    ptr = :(Base.unsafe_convert(DevicePtr{UInt8,AS.Global}, $(esc(oc)).hostcall.buf_ptr))
    push!(ex.args, :(Base.unsafe_store!($ptr, $byte, $N)))
    return N+1
end
function rocprint!(ex, N, oc, iex::Expr)
    for arg in iex.args
        N = rocprint!(ex, N, oc, arg)
    end
    return N
end
function rocprint!(ex, N, oc, sym::S) where S
    error("Dynamic printing of $S only supported via @rocprintf")
end

## @rocprintf

macro rocprintf(fmt, args...)
    ex = Expr(:block)
    @gensym device_ptr device_fmt_ptr printf_hc
    push!(ex.args, :($device_fmt_ptr = AMDGPU.alloc_string($(Val(Symbol(fmt))))))
    push!(ex.args, :($printf_hc = unsafe_load(AMDGPU.get_global_pointer(Val(:__global_printf_context),
                                                                        HostCall{UInt64,Int64,DevicePtr{ROCPrintfBuffer,AS.Global}}))))
    push!(ex.args, :($device_ptr = $printf_hc.buf_ptr))
    push!(ex.args, :($device_ptr = AMDGPU._rocprintf_fmt($device_ptr, $device_fmt_ptr, $(sizeof(fmt)))))
    for arg in args
        push!(ex.args, :($device_ptr = AMDGPU._rocprintf_arg($device_ptr, $arg)))
    end
    push!(ex.args, :(AMDGPU.hostcall!($printf_hc, ROCPrintfBuffer())))
    ex
end

macro rocprintfw(fmt, args...)
    quote
        AMDGPU.wave_serialized() do
            @rocprintf($fmt, $(args...))
        end
    end
end

# Serializes execution of a function within a wavefront
# From implementation by @jonathanvdc in CUDAnative.jl#419
function wave_serialized(func::Function)
    # Get the current thread's ID
    thread_id = workitemIdx().x - 1

    # Get the size of a wavefront
    size = wavefrontsize()

    local result
    i = 0
    while i < size
        if thread_id % size == i
            result = func()
        end
        i += 1
    end
    return result
end

struct ROCPrintfBuffer end
Base.sizeof(::ROCPrintfBuffer) = 0
Base.unsafe_store!(::DevicePtr{ROCPrintfBuffer,as} where as, x) = nothing
function Base.unsafe_load(ptr::DevicePtr{ROCPrintfBuffer,as} where as)
    ptr = Base.unsafe_convert(DevicePtr{UInt8,AS.Global}, ptr)
    fmt = Base.unsafe_string(convert(Ptr{UInt8}, ptr))
    ptr += sizeof(fmt)+1
    args = []
    while true
        name = Base.unsafe_string(convert(Ptr{UInt8}, ptr))
        name == "" && break
        T = eval(Meta.parse(name))
        ptr += sizeof(name)+1
        arg = unsafe_load(Base.unsafe_convert(DevicePtr{T,AS.Global}, ptr))
        push!(args, arg)
        ptr += sizeof(arg)
    end
    return (fmt, args)
end
function _rocprintf_fmt(ptr, fmt_ptr, fmt_len)
    AMDGPU.memcpy!(ptr, fmt_ptr, fmt_len)
    unsafe_store!(ptr+fmt_len, UInt8(0))
    return ptr+fmt_len+1
end
function _rocprintf_arg(ptr, arg::T) where T
    T_str, T_str_len = _rocprintf_T_str(T)
    AMDGPU.memcpy!(ptr, T_str, T_str_len)
    ptr += T_str_len
    unsafe_store!(ptr, UInt8(0))
    ptr += 1
    unsafe_store!(Base.unsafe_convert(DevicePtr{T,AS.Global}, ptr), arg)
    ptr += sizeof(arg)
    return ptr
end
#= TODO: Not really useful until we can work with device-side strings
function _rocprintf_string(ptr, str::String)
    @gensym T_str T_str_len str_ptr
    quote
        $T_str, $T_str_len = AMDGPU._rocprintf_T_str(String)
        AMDGPU.memcpy!($ptr, $T_str, $T_str_len)
        $ptr += $T_str_len
        unsafe_store!($ptr, UInt8(0))
        $ptr += 1
        $str_ptr = Base.unsafe_convert(DevicePtr{UInt8,AS.Generic}, $str_ptr)
        $str_ptr = AMDGPU.alloc_string($(Val(Symbol(str))))
        AMDGPU.memcpy!($ptr, $str_ptr, $(length(str)))
        $ptr += $(length(str))
        $ptr
    end
end
=#
@generated function _rocprintf_T_str(::Type{T}) where T
    quote
        (AMDGPU.alloc_string($(Val(Symbol(repr(T))))), $(sizeof(repr(T))))
    end
end
