# TODO
# - serial version for lower latency
# - block-stride loop to delay need for second kernel launch

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op, val::T, neutral, wfops::Val{true}) where T
    shared = @alloc_special(AS.Local, T, 64)  # NOTE: this is an upper bound; better detect it
    #@rocprintln "In reduce_block"

    idx = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    #idx = workitemIdx().x
    wid, lane = (fld1(idx, 64), mod1(idx, 64))

    # each wavefront performs partial reduction
    val = wfred(op, val)

    # write reduced value to shared memory
    if lane == 1
        #=@inbounds=# shared[wid] = val
    end

    # wait for all partial reductions
    sync_workgroup()

    # read from shared memory only if that wavefront existed
    val = if workitemIdx().x <= fld1(workgroupDim().x, 64)
         #=@inbounds=# shared[lane]
         #1.0
    else
        neutral
    end

    # final reduce within first wavefront
    if wid == 1
        val = wfred(op, val)
    end
    return val
end
#= FIXME
@inline function reduce_block(op, val::T, neutral, wfops::Val{false}) where T
    threads = workgroupDim().x
    thread = workitemIdx().x

    # shared mem for a complete reduction
    shared = @alloc_special(AS.Local, T, (2*threads,))
    #=@inbounds=# shared[thread] = val

    # perform a reduction
    d = threads>>1
    while d > 0
        sync_workgroup()
        if thread <= d
            shared[thread] = op(shared[thread], shared[thread+d])
        end
        d >>= 1
    end

    # load the final value on the first thread
    if thread == 1
        val = #=@inbounds=# shared[thread]
    end

    return val
end
=#

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
function partial_mapreduce_grid(f, op, neutral, Rreduce, Rother, wfops, R, As...)
    # decompose the 1D hardware indices into separate ones for reduction (across threads
    # and possibly blocks if it doesn't fit) and other elements (remaining blocks)
    workitemIdx_reduce = workitemIdx().x
    workgroupDim_reduce = workgroupDim().x
    #workgroupIdx_reduce, workgroupIdx_other = fldmod1(workgroupIdx().x, length(Rother))
    workgroupIdx_reduce, workgroupIdx_other = (fld1(workgroupIdx().x, length(Rother)), mod1(workgroupIdx().x, length(Rother)))
    gridDim_reduce = gridDimWG().x ÷ length(Rother)

    # block-based indexing into the values outside of the reduction dimension
    # (that means we can safely synchronize threads within this block)
    iother = workgroupIdx_other
    #=@inbounds=# if iother <= length(Rother)
        Iother = Rother[iother]

        # load the neutral value
        Iout = CartesianIndex(Tuple(Iother)..., workgroupIdx_reduce)
        neutral = if neutral === nothing
            R[Iout]
        else
            neutral
        end

        val = op(neutral, neutral)

        # reduce serially across chunks of input vector that don't fit in a block
        ireduce = workitemIdx_reduce + (workgroupIdx_reduce - 1) * workgroupDim_reduce
        while ireduce <= length(Rreduce)
            Ireduce = Rreduce[ireduce]
            J = Base.max(Iother, Ireduce)
            val = op(val, f(_map_getindex(As, J)...))
            ireduce += workgroupDim_reduce * gridDim_reduce
        end

        val = reduce_block(op, val, neutral, wfops)

        # write back to memory
        if workitemIdx_reduce == 1
            R[Iout] = val
        end
    end

    return
end

## COV_EXCL_STOP

if VERSION < v"1.5.0-DEV.748"
    Base.axes(bc::Base.Broadcast.Broadcasted{<:ROCArrayStyle, <:NTuple{N}},
              d::Integer) where N =
        d <= N ? axes(bc)[d] : Base.OneTo(1)
end

function GPUArrays.mapreducedim!(f::F, op::OP, R::ROCArray{T},
                                 A::Union{AbstractArray,Broadcast.Broadcasted};
                                 init=nothing) where {F, OP, T}
    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    f = rocfunc(f)
    op = rocfunc(op)

    # be conservative about using wavefront intrinsics
    wfops = ((OP in (typeof(+), typeof(max), typeof(min))) &&
             (T <: Union{Cint, Cuint, Clong, Culong, Float32, Float64})) ||
            ((OP in (typeof(&), typeof(|), typeof(⊻))) &&
             (T <: Union{Cint, Cuint, Clong, Culong}))

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)

    # allocate an additional, empty dimension to write the reduced value to.
    # this does not affect the actual location in memory of the final values,
    # but allows us to write a generalized kernel supporting partial reductions.
    R′ = reshape(R, (size(R)..., 1))

    # how many threads do we want?
    #
    # threads in a block work together to reduce values across the reduction dimensions;
    # we want as many as possible to improve algorithm efficiency and execution occupancy.
    agent = R.buf.agent
    wanted_threads = wfops ? nextwavefront(agent, length(Rreduce)) : nextpow(2, length(Rreduce))
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            wfops ? prevwavefront(agent, max_threads) : prevpow(2, max_threads)
        else
            wanted_threads
        end
    end

    # how many threads can we launch?
    #
    # we might not be able to launch all those threads to reduce each slice in one go.
    # that's why each threads also loops across their inputs, processing multiple values
    # so that we can span the entire reduction dimension using a single thread block.
    args = (f, op, init, Rreduce, Rother, Val(wfops), R′, A)
    kernel_args = rocconvert.(args)
    kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
    kernel = rocfunction(partial_mapreduce_grid, kernel_tt)
    #= TODO:
    compute_shmem(threads) = wfops ? 0 : 2*threads*sizeof(T)
    kernel_config = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
    reduce_threads = compute_threads(kernel_config.threads)
    reduce_shmem = compute_shmem(reduce_threads)
    =#
    reduce_threads = wanted_threads
    reduce_shmem = 0
    kernel_config = (blocks=1,)

    # how many blocks should we launch?
    #
    # even though we can always reduce each slice in a single thread block, that may not be
    # optimal as it might not saturate the GPU. we already launch some blocks to process
    # independent dimensions in parallel; pad that number to ensure full occupancy.
    other_blocks = length(Rother)
    reduce_blocks = if other_blocks >= kernel_config.blocks
        1
    else
        Base.min(cld(length(Rreduce), reduce_threads),       # how many we need at most
                 cld(kernel_config.blocks, other_blocks))    # maximize occupancy
    end

    # determine the launch configuration
    threads = reduce_threads
    shmem = reduce_shmem
    blocks = reduce_blocks*other_blocks
    blocks = 1 # FIXME

    # perform the actual reduction
    if reduce_blocks == 1
        @show (threads, blocks)
        @show (f, op, init, Rreduce, Rother)
        @show R′
        @show A
        # we can cover the dimensions to reduce using a single block
        wait(@roc groupsize=threads gridsize=blocks*threads partial_mapreduce_grid(
            f, op, init, Rreduce, Rother, Val(wfops), R′, A))
    else
        # we need multiple steps to cover all values to reduce
        partial = similar(R, (size(R)..., reduce_blocks))
        if init === nothing
            # without an explicit initializer we need to copy from the output container
            sz = prod(size(R))
            for i in 1:reduce_blocks
                # TODO: async copies (or async fill!, but then we'd need to load first)
                #       or maybe just broadcast since that extends singleton dimensions
                copyto!(partial, (i-1)*sz+1, R, 1, sz)
            end
        end
        wait(@roc groupsize=threads gridsize=blocks*threads partial_mapreduce_grid(
            f, op, init, Rreduce, Rother, Val(wfops), partial, A))

        GPUArrays.mapreducedim!(identity, op, R′, partial; init=init)
    end

    return R
end
