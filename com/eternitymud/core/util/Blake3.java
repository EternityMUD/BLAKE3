/*
 * NOTICE:
 * <p>
 * Pure Java(tm) implementation of BLAKE3 cryptographic hash function.
 * <p>
 * Main features:
 * - Extends MessageDigest class and includes Provider method.
 * - SIMD and scalar compressor modes with automatic fallback.
 * - SIMD vectorization of blocks, chunks and parallel chunks.
 * - Multi-threading in SIMD and scalar compression of chunks.
 * - Configurable: enable or disable SIMD and multi-threading.
 * - SIMD lanes and number of threads scale with the hardware.
 * <p>
 * This implementation was developed by Archon Research for internal use in
 * EternityMUD, but sadly did not meet the performance requirements and was
 * abandoned at least for now. We are releasing it as open source software.
 * <p>
 * You are free to use this software in your products, but you must include
 * a statement that you are using software developed by Archon Research for
 * EternityMUD and a link to: https://www.eternitymud.com
 * <p>
 * DISCLAIMER: this is free and open source software. The developer takes no
 * responsibility for any personal, material or other damage the use of this
 * software might cause.
 * <p>
 * This software was developed and tested using Java 25.
 * <p>
 * Only human blood, sweat, tears and TIME were wasted during the development.
 * AI tools have not been used during the development of this software.
 * <p>
 * Original API documentation is available here:
 * https://www.eternitymud.com/doc/java/com/eternitymud/core/util/Blake3.html
 */
package com.eternitymud.core.util;
import java.io.Closeable;
import java.io.IOException;
import java.lang.ref.Reference;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.DigestException;
import java.security.MessageDigest;
import java.security.MessageDigestSpi;
import java.security.NoSuchAlgorithmException;
import java.security.Provider;
import java.security.Provider.Service;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/**
 * Pure Java&trade; implementation of <a href="https://github.com/BLAKE3-team/BLAKE3">BLAKE3</a> cryptographic hash
 * function with SIMD vectorization. The vectorization API is an incubator feature as of Java 25. The following Java
 * virtual machine launch options are required to enable the feature:
 * <p>
 * <code>&nbsp;&nbsp;&nbsp;&nbsp;--enable-preview<br>&nbsp;&nbsp;&nbsp;&nbsp;--add-modules=jdk.incubator.vector</code>
 * <p>
 * This class implements two types of SIMD (Single Instruction Multiple Data) parallelism that are discussed in section
 * 5.3 of BLAKE3 <a href="https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf">whitepaper</a>. Individual
 * blocks and a small number of chunks are compressed in serial using fixed-width 128-bit vectors. Large number of
 * chunks are compressed in parallel using a variable number of vector lanes that is scaled based on hardware. Scalar
 * compressor is used {@link Blake3#simd() automatically} if the vectorization API is not available on the runtime.
 * <p>
 * Parallel threads are used when compressing multiple chunks. The compressor uses half of the available cores at
 * maximum while the number of concurrent threads scales with the hardware. Multi-threading is supported when using SIMD
 * or scalar compressor alike. A small number of chunks are compressed sequentially.
 * <p>
 * SIMD parallelism and multi-threading performance depends on the AVX/NEON etc. capabilities of the hardware. If the
 * runtime has 32 cores and AVX-512 instruction set then at most 16 threads are used to compress 16 chunks in parallel
 * by each concurrent thread for a total of 256 chunks.
 * <p>
 * Instances of this class are designed to be used like {@link MessageDigest} objects. Incremental updates to the
 * hash value are supported, but the hash state is {@link Blake3#reset() reset} when a {@link Blake3#digest(int) digest}
 * is computed. The instance can then be used again to compute another hash value. The instance can be
 * {@link Blake3#clone() cloned} if many states of a hash must be preserved or digests of different lengths from the
 * same hash are needed. See the {@link Blake3#provider() provider} method for more information.
 * <p>
 * Example:
 * <p>
 * <code>
 * &nbsp;&nbsp;&nbsp;&nbsp;Blake3 b1=new Blake3();<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;b1.update("Hello there".getBytes(StandardCharsets.UTF_8));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;Blake3 b2=b1.clone();<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;b2.update(" world!".getBytes(StandardCharsets.UTF_8));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;System.out.println(Base64.getEncoder().encodeToString(b1.digest()));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;-&gt; 9FQM6IvkXlqP1tGY/5Dax+s03Yx51t+4dnoXALEIrm8=<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;System.out.println(Base64.getEncoder().encodeToString(b2.digest()));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;-&gt; 6XeH1EeHk0/YYn+x73TYgOHk9Y+ibl8Z0d56OOXH4mk=<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;b1.update("Hello again!".getBytes(StandardCharsets.UTF_8));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;System.out.println(Base64.getEncoder().encodeToString(b1.digest()));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;-&gt; PyX4XKZIhebsMQd88DHHBUrMnl99D7HzdblCEFtcxAs=
 * </code>
 * <p>
 * It is highly recommended to reuse hasher instances as long as they are needed to minimize the potential overhead of
 * internal thread management, unless it is required not to do so. If multiple concurrent threads share a single hasher
 * instance then it <i>must</i> be synchronized externally.
 * <p>
 * There are two system properties that can be used to configure Blake3 instances:
 * <p>
 * <code>
 * &nbsp;&nbsp;&nbsp;&nbsp;com.eternitymud.core.util.Blake3.simd=true<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;com.eternitymud.core.util.Blake3.threads=true
 * </code>
 * <p>
 * If those properties are set to <code>false</code> then SIMD vectorization or multi-threading is disabled. The
 * properties must be set before this class is loaded, usually when Java virtual machine is started.
 * <p>
 * This implementation has been tested with the
 * <a href="https://github.com/BLAKE3-team/BLAKE3/blob/master/test_vectors/test_vectors.json">vectors</a> of the
 * official BLAKE3 repository for all types (regular hash, keyed hash, key derivation) with and without SIMD
 * vectorization. Numerous files of different sizes have also been hashed and the output matched to
 * <a href="https://crates.io/crates/b3sum"><code>b3sum</code></a> utility.
 *
 * @version 1.0
 * @author Zan the Archon &lt;zan@fantasymail.de&gt;
 */
public final class Blake3 extends MessageDigest implements Cloneable
{
	private static final class Blake3Provider extends Provider
	{
		private static final long serialVersionUID=1L;

		private Blake3Provider()
		{
			super("Blake3Provider","1.0","Blake3Provider - Blake3 message digest provider");
			final String type="MessageDigest";
			final String name=Blake3.class.getCanonicalName();
			putService(new Blake3Service(this,type,Blake3.ALGORITHM,name,null,null));
			putService(new Blake3Service(this,type,Blake3.ALGORITHM_SCALAR,name,null,null));
			putService(new Blake3Service(this,type,Blake3.ALGORITHM_SIMD,name,null,null));
		}
	}

	private static final class Blake3Service extends Service
	{
		public Blake3Service(final Provider provider, final String type, final String algorithm, final String className, final List <String> aliases, final Map <String, String> attributes)
		{
			super(provider,type,algorithm,className,aliases,attributes);
		}

		@Override
		public Blake3 newInstance(final Object constructorParameter) throws NoSuchAlgorithmException
		{
			return new Blake3(getAlgorithm());
		}

		@Override
		public boolean supportsParameter(final Object parameter)
		{
			return false;
		}
	}

	private final class ChunkState implements Cloneable
	{
		private static final byte ZERO=0;
		private final Output out;
		private final byte[] block;
		private final int[] state;
		private final int[] words;
		private final int[] chainingValue;
		private long counter;
		private int length;
		private int blocks;
		private final int flags;

		private ChunkState(final ChunkState c)
		{
			out=c.out.clone();
			block=c.block.clone();
			state=c.state.clone();
			words=c.words.clone();
			chainingValue=c.chainingValue.clone();
			counter=c.counter;
			length=c.length;
			blocks=c.blocks;
			flags=c.flags;
		}

		private ChunkState(final int[] key, final int flags)
		{
			out=new Output();
			block=new byte[Blake3.BLOCK_LEN];
			state=new int[Blake3.WORDS_LEN];
			words=new int[Blake3.WORDS_LEN];
			chainingValue=key.clone();
			counter=length=blocks=0;
			this.flags=flags;
		}

		private void clear(final int[] key)
		{
			reset(key);
			counter=0;
		}

		@Override
		public ChunkState clone()
		{
			return new ChunkState(this);
		}

		private int length()
		{
			return Blake3.BLOCK_LEN*blocks+length;
		}

		private Output output()
		{
			return out.set(state,Blake3.words(block,words),chainingValue,counter,length,startFlag(flags|Blake3.CHUNK_END));
		}

		private void reset(final int[] key)
		{
			System.arraycopy(key,0,chainingValue,0,Blake3.CHAIN_LEN);
			Arrays.fill(block,ChunkState.ZERO);
			length=blocks=0;
		}

		private int startFlag(final int flags)
		{
			return blocks==0?flags|Blake3.CHUNK_START:flags;
		}

		private void update(final byte[] input, int off, final int limit, final Compressor c)
		{
			int n;
			while(off<limit)
			{
				if(length==Blake3.BLOCK_LEN)
				{
					/*
					 * After rather extensive profiling it was found that the hasher spends most of its time here
					 * compressing blocks, specifically doing mixer function G rounds. If hashing some input file
					 * takes 3s then about 2s is spent here. It is difficult to find a way to optimize this. G is
					 * already virtually as optimal as possible in the SIMD and scalar compressor methods. Thread
					 * parallelism doesn't help, since this cannot be computed in parallel. Rust and C seem to be
					 * far superior to Java performing the arithmetic operations of the mixer function. Arena and
					 * MemorySegment objects were tested, but they resulted in horrible performance where 3s task
					 * took 8-10s to compute with SIMD compressor. Unsafe is deprecated for removal and therefore
					 * optimization ideas for this are few. TLDR: the bottleneck of Blake3 in Java is not SIMD or
					 * thread parallelism but this compression of blocks. Java intrinsics are very likely needed.
					 */
					c.compress(state,Blake3.words(block,words),chainingValue,counter,length,startFlag(flags));
					System.arraycopy(state,0,chainingValue,0,Blake3.CHAIN_LEN);
					Arrays.fill(block,ChunkState.ZERO);
					length=0;
					blocks++;
				}
				System.arraycopy(input,off,block,length,n=Math.min(Blake3.BLOCK_LEN-length,limit-off));
				length+=n;
				off+=n;
			}
		}
	}

	private abstract static class Compressor implements Cloneable
	{
		private static final int ROUNDS=7;
		private static final int[][] SCHEDULE={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},{2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},{3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},{10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},{12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},{9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},{11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}};

		private Compressor()
		{
			super();
		}

		@Override
		public abstract Compressor clone();

		final int[] compress(final int[] state, final int[] words, final int[] chainingValue, final long counter, final int length, final int flags)
		{
			return function(update(state,chainingValue,counter,length,flags),words,chainingValue);
		}

		abstract Collection <int[]> compress(List <ChunkState> chunks, int start, int end);

		abstract int[] function(int[] state, int[] words, int[] chainingValue);

		abstract int parallelism();

		abstract boolean simd();

		final int[] update(final int[] state, final int[] chainingValue, final long counter, final int length, final int flags)
		{
			System.arraycopy(chainingValue,0,state,0,Blake3.CHAIN_LEN);
			System.arraycopy(Blake3.IV,0,state,Blake3.CHAIN_LEN,4);
			state[12]=(int)counter;
			state[13]=(int)(counter>>32);
			state[14]=length;
			state[15]=flags;
			return state;
		}
	}

	private final class Output implements Cloneable
	{
		private int[] state;
		private int[] words;
		private int[] chainingValue;
		private long counter;
		private int length;
		private int flags;

		private Output()
		{
			state=words=chainingValue=null;
			counter=length=flags=0;
		}

		private Output(final Output o)
		{
			state=o.state!=null?o.state.clone():null;
			words=o.words!=null?o.words.clone():null;
			chainingValue=o.chainingValue!=null?o.chainingValue.clone():null;
			counter=o.counter;
			length=o.length;
			flags=o.flags;
		}

		private int[] chainingValue(final Compressor c)
		{
			return c.compress(state,words,chainingValue,counter,length,flags);
		}

		@Override
		public Output clone()
		{
			return new Output(this);
		}

		private Output parentOutput(final int[] state, final int[] leftLeaf, final int[] rightLeaf, final int[] chainingValue, final int flags)
		{
			System.arraycopy(rightLeaf,0,leftLeaf,Blake3.CHAIN_LEN,Blake3.CHAIN_LEN);
			return set(state,leftLeaf,chainingValue,0,Blake3.BLOCK_LEN,flags|Blake3.PARENT);
		}

		private byte[] rootBytes(final byte[] b, int off, final int len, final Compressor c)
		{
			final int root=flags|Blake3.ROOT;
			final int limit=Math.addExact(off,len);
			for(int z=off, i, j, k; z<limit; off++, z+=Blake3.BLOCK_LEN)
			{
				c.compress(state,words,chainingValue,off,length,root);
				final int len2=Math.min(limit-z+3,Blake3.BLOCK_LEN)/Integer.BYTES;
				for(i=0, j=i; i<len2; i++, j+=Integer.BYTES)
				{
					final long l=i<state.length?state[i]:0;
					b[k=z+j]=(byte)l;
					if(++k<b.length)
					{
						b[k]=(byte)(l>>8);
						if(++k<b.length)
						{
							b[k]=(byte)(l>>16);
							if(++k<b.length)
								b[k]=(byte)(l>>24);
						}
					}
				}
			}
			return b;
		}

		private Output set(final int[] state, final int[] words, final int[] chainingValue, final long counter, final int length, final int flags)
		{
			this.state=state;
			this.words=words;
			this.chainingValue=chainingValue;
			this.counter=counter;
			this.length=length;
			this.flags=flags;
			return this;
		}
	}

	private static final class Scalar extends Compressor
	{
		private static final int P=Math.max(1,Runtime.getRuntime().availableProcessors()/2);

		private static void g(final int[] state, final int a, final int b, final int c, final int d, final int x, final int y)
		{
			state[a]+=state[b]+x;
			state[d]=Integer.rotateRight(state[d]^state[a],16);
			state[c]+=state[d];
			state[b]=Integer.rotateRight(state[b]^state[c],12);
			state[a]+=state[b]+y;
			state[d]=Integer.rotateRight(state[d]^state[a],8);
			state[c]+=state[d];
			state[b]=Integer.rotateRight(state[b]^state[c],7);
		}

		private static void round(final int[] state, final int[] m, final int[] s)
		{
			Scalar.g(state,0,4,8,12,m[s[0]],m[s[1]]);
			Scalar.g(state,1,5,9,13,m[s[2]],m[s[3]]);
			Scalar.g(state,2,6,10,14,m[s[4]],m[s[5]]);
			Scalar.g(state,3,7,11,15,m[s[6]],m[s[7]]);
			Scalar.g(state,0,5,10,15,m[s[8]],m[s[9]]);
			Scalar.g(state,1,6,11,12,m[s[10]],m[s[11]]);
			Scalar.g(state,2,7,8,13,m[s[12]],m[s[13]]);
			Scalar.g(state,3,4,9,14,m[s[14]],m[s[15]]);
		}

		private Scalar()
		{
			super();
		}

		@Override
		public Compressor clone()
		{
			return this;
		}

		@Override
		Collection <int[]> compress(final List <ChunkState> chunks, int start, final int end)
		{
			final Queue <int[]> q=new ArrayDeque <>(end-start);
			do
			{
				q.add(chunks.get(start).output().chainingValue(this));
			}
			while(++start<end);
			return q;
		}

		@Override
		int[] function(final int[] state, final int[] words, final int[] chainingValue)
		{
			int i=0, j=i;
			do
			{
				Scalar.round(state,words,Compressor.SCHEDULE[i++]);
			}
			while(i<Compressor.ROUNDS);
			do
			{
				state[j]^=state[++i];
				state[i]^=chainingValue[j++];
			}
			while(j<Blake3.CHAIN_LEN);
			return state;
		}

		@Override
		int parallelism()
		{
			return Blake3.THREADS_ENABLED?Scalar.P:1;
		}

		@Override
		boolean simd()
		{
			return false;
		}
	}

	private static final class Simd extends Compressor
	{
		/** Vector species used to compress multiple chunks in parallel. */
		private static final VectorSpecies <Integer> PREF=IntVector.SPECIES_PREFERRED;
		/** Vector species used to compress blocks or individual chunks. */
		private static final VectorSpecies <Integer> SPEC=IntVector.SPECIES_128;
		private static final VectorShuffle <Integer> S1=VectorShuffle.fromArray(Simd.SPEC,new int[]{1,2,3,0},0);
		private static final VectorShuffle <Integer> S2=VectorShuffle.fromArray(Simd.SPEC,new int[]{2,3,0,1},0);
		private static final VectorShuffle <Integer> S3=VectorShuffle.fromArray(Simd.SPEC,new int[]{3,0,1,2},0);
		private static final int P=Runtime.getRuntime().availableProcessors()/2<2?1:Simd.PREF.length();

		private Simd()
		{
			super();
		}

		@Override
		public Compressor clone()
		{
			return this;
		}

		@Override
		Collection <int[]> compress(final List <ChunkState> chunks, final int start, final int end)
		{
			/*
			 * Java vectors are value-based classes, and cannot be used as array elements or method parameters without
			 * extreme performance penalties. All compressor rounds and the mixer function G are done directly in this
			 * method because of that property.
			 */
			int i;
			final int lanes=end-start;
			final Queue <int[]> states;
			final Queue <int[]> words;
			final Queue <int[]> chainingValues;
			{
				i=start;
				final List <int[]> s=new ArrayList <>(lanes);
				final List <int[]> w=new ArrayList <>(lanes);
				final List <int[]> c=new ArrayList <>(lanes);
				do
				{
					final Output o=chunks.get(i).output();
					s.add(update(o.state,o.chainingValue,o.counter,o.length,o.flags));
					w.add(o.words);
					c.add(o.chainingValue);
				}
				while(++i<end);
				states=transpose(s,Blake3.WORDS_LEN,lanes);
				words=transpose(w,Blake3.WORDS_LEN,lanes);
				chainingValues=transpose(c,Blake3.CHAIN_LEN,lanes);
			}
			IntVector v0=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v1=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v2=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v3=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v4=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v5=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v6=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v7=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v8=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v9=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v10=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v11=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v12=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v13=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v14=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector v15=IntVector.fromArray(Simd.PREF,states.poll(),0);
			IntVector a=null, b=a, c=a, d=a, x=a, y=a;
			final int[][] m=matrix(words);
			int j=i=0;
			do
			{
				final int[] s=Compressor.SCHEDULE[i++];
				do
				{
					switch(j)
					{
						case 0:
							a=v0;
							b=v4;
							c=v8;
							d=v12;
							x=IntVector.fromArray(Simd.PREF,m[s[0]],0);
							y=IntVector.fromArray(Simd.PREF,m[s[1]],0);
							break;
						case 1:
							a=v1;
							b=v5;
							c=v9;
							d=v13;
							x=IntVector.fromArray(Simd.PREF,m[s[2]],0);
							y=IntVector.fromArray(Simd.PREF,m[s[3]],0);
							break;
						case 2:
							a=v2;
							b=v6;
							c=v10;
							d=v14;
							x=IntVector.fromArray(Simd.PREF,m[s[4]],0);
							y=IntVector.fromArray(Simd.PREF,m[s[5]],0);
							break;
						case 3:
							a=v3;
							b=v7;
							c=v11;
							d=v15;
							x=IntVector.fromArray(Simd.PREF,m[s[6]],0);
							y=IntVector.fromArray(Simd.PREF,m[s[7]],0);
							break;
						case 4:
							a=v0;
							b=v5;
							c=v10;
							d=v15;
							x=IntVector.fromArray(Simd.PREF,m[s[8]],0);
							y=IntVector.fromArray(Simd.PREF,m[s[9]],0);
							break;
						case 5:
							a=v1;
							b=v6;
							c=v11;
							d=v12;
							x=IntVector.fromArray(Simd.PREF,m[s[10]],0);
							y=IntVector.fromArray(Simd.PREF,m[s[11]],0);
							break;
						case 6:
							a=v2;
							b=v7;
							c=v8;
							d=v13;
							x=IntVector.fromArray(Simd.PREF,m[s[12]],0);
							y=IntVector.fromArray(Simd.PREF,m[s[13]],0);
							break;
						case 7:
							a=v3;
							b=v4;
							c=v9;
							d=v14;
							x=IntVector.fromArray(Simd.PREF,m[s[14]],0);
							y=IntVector.fromArray(Simd.PREF,m[s[15]],0);
							break;
					}
					a=a.lanewise(VectorOperators.ADD,b).lanewise(VectorOperators.ADD,x);
					d=d.lanewise(VectorOperators.XOR,a).lanewise(VectorOperators.ROR,16);
					c=c.lanewise(VectorOperators.ADD,d);
					b=b.lanewise(VectorOperators.XOR,c).lanewise(VectorOperators.ROR,12);
					a=a.lanewise(VectorOperators.ADD,b).lanewise(VectorOperators.ADD,y);
					d=d.lanewise(VectorOperators.XOR,a).lanewise(VectorOperators.ROR,8);
					c=c.lanewise(VectorOperators.ADD,d);
					b=b.lanewise(VectorOperators.XOR,c).lanewise(VectorOperators.ROR,7);
					switch(j)
					{
						case 0:
							v0=a;
							v4=b;
							v8=c;
							v12=d;
							break;
						case 1:
							v1=a;
							v5=b;
							v9=c;
							v13=d;
							break;
						case 2:
							v2=a;
							v6=b;
							v10=c;
							v14=d;
							break;
						case 3:
							v3=a;
							v7=b;
							v11=c;
							v15=d;
							break;
						case 4:
							v0=a;
							v5=b;
							v10=c;
							v15=d;
							break;
						case 5:
							v1=a;
							v6=b;
							v11=c;
							v12=d;
							break;
						case 6:
							v2=a;
							v7=b;
							v8=c;
							v13=d;
							break;
						case 7:
							v3=a;
							v4=b;
							v9=c;
							v14=d;
							break;
					}
				}
				while(++j<Blake3.CHAIN_LEN);
				j=0;
			}
			while(i<Compressor.ROUNDS);
			v0=v0.lanewise(VectorOperators.XOR,v8);
			v8=v8.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.PREF,chainingValues.poll(),0));
			v1=v1.lanewise(VectorOperators.XOR,v9);
			v9=v9.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.PREF,chainingValues.poll(),0));
			v2=v2.lanewise(VectorOperators.XOR,v10);
			v10=v10.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.PREF,chainingValues.poll(),0));
			v3=v3.lanewise(VectorOperators.XOR,v11);
			v11=v11.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.PREF,chainingValues.poll(),0));
			v4=v4.lanewise(VectorOperators.XOR,v12);
			v12=v12.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.PREF,chainingValues.poll(),0));
			v5=v5.lanewise(VectorOperators.XOR,v13);
			v13=v13.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.PREF,chainingValues.poll(),0));
			v6=v6.lanewise(VectorOperators.XOR,v14);
			v14=v14.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.PREF,chainingValues.poll(),0));
			v7=v7.lanewise(VectorOperators.XOR,v15);
			v15=v15.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.PREF,chainingValues.poll(),0));
			final List <int[]> l=new ArrayList <>(Blake3.WORDS_LEN);
			{
				l.add(v0.toArray());
				l.add(v1.toArray());
				l.add(v2.toArray());
				l.add(v3.toArray());
				l.add(v4.toArray());
				l.add(v5.toArray());
				l.add(v6.toArray());
				l.add(v7.toArray());
				l.add(v8.toArray());
				l.add(v9.toArray());
				l.add(v10.toArray());
				l.add(v11.toArray());
				l.add(v12.toArray());
				l.add(v13.toArray());
				l.add(v14.toArray());
				l.add(v15.toArray());
			}
			return transpose(l,lanes,Blake3.WORDS_LEN);
		}

		@Override
		int[] function(final int[] state, final int[] words, final int[] chainingValue)
		{
			/*
			 * Java vectors are value-based classes, and cannot be used as array elements or method parameters without
			 * extreme performance penalties. All compressor rounds and the mixer function G are done directly in this
			 * method because of that property.
			 */
			final int[] x=new int[Blake3.CHAIN_LEN];
			final int[] y=new int[Blake3.CHAIN_LEN];
			IntVector a=IntVector.fromArray(Simd.SPEC,state,0);
			IntVector b=IntVector.fromArray(Simd.SPEC,state,4);
			IntVector c=IntVector.fromArray(Simd.SPEC,state,8);
			IntVector d=IntVector.fromArray(Simd.SPEC,state,12);
			int i=0;
			do
			{
				permute(words,x,y,Compressor.SCHEDULE[i++]);
				a=a.lanewise(VectorOperators.ADD,b).lanewise(VectorOperators.ADD,IntVector.fromArray(Simd.SPEC,x,0));
				d=d.lanewise(VectorOperators.XOR,a).lanewise(VectorOperators.ROR,16);
				c=c.lanewise(VectorOperators.ADD,d);
				b=b.lanewise(VectorOperators.XOR,c).lanewise(VectorOperators.ROR,12);
				a=a.lanewise(VectorOperators.ADD,b).lanewise(VectorOperators.ADD,IntVector.fromArray(Simd.SPEC,y,0));
				d=d.lanewise(VectorOperators.XOR,a).lanewise(VectorOperators.ROR,8);
				c=c.lanewise(VectorOperators.ADD,d);
				b=b.lanewise(VectorOperators.XOR,c).lanewise(VectorOperators.ROR,7);
				b=b.rearrange(Simd.S1);
				c=c.rearrange(Simd.S2);
				d=d.rearrange(Simd.S3);
				a=a.lanewise(VectorOperators.ADD,b).lanewise(VectorOperators.ADD,IntVector.fromArray(Simd.SPEC,x,4));
				d=d.lanewise(VectorOperators.XOR,a).lanewise(VectorOperators.ROR,16);
				c=c.lanewise(VectorOperators.ADD,d);
				b=b.lanewise(VectorOperators.XOR,c).lanewise(VectorOperators.ROR,12);
				a=a.lanewise(VectorOperators.ADD,b).lanewise(VectorOperators.ADD,IntVector.fromArray(Simd.SPEC,y,4));
				d=d.lanewise(VectorOperators.XOR,a).lanewise(VectorOperators.ROR,8);
				c=c.lanewise(VectorOperators.ADD,d);
				b=b.lanewise(VectorOperators.XOR,c).lanewise(VectorOperators.ROR,7);
				b=b.rearrange(Simd.S3);
				c=c.rearrange(Simd.S2);
				d=d.rearrange(Simd.S1);
			}
			while(i<Compressor.ROUNDS);
			a.lanewise(VectorOperators.XOR,c).intoArray(state,0);
			b.lanewise(VectorOperators.XOR,d).intoArray(state,4);
			c.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.SPEC,chainingValue,0)).intoArray(state,8);
			d.lanewise(VectorOperators.XOR,IntVector.fromArray(Simd.SPEC,chainingValue,4)).intoArray(state,12);
			return state;
		}

		private int[][] matrix(final Queue <int[]> q)
		{
			final int[][] m=new int[q.size()][];
			{
				for(int i=0; i<m.length; i++)
					m[i]=q.poll();
			}
			return m;
		}

		@Override
		int parallelism()
		{
			/*
			 * SIMD always tries to compress as many chunks at a time as possible in a single thread or parallel
			 * threads. This method can return 1 to disable bundle compression of chunks and thread parallelism.
			 * For example: return Blake3.THREADS_ENABLED?Simd.P:1;
			 */
			return Simd.P;
		}

		private void permute(final int[] m, final int[] x, final int[] y, final int[] s)
		{
			int i=0, j=i;
			do
			{
				x[j]=m[s[i++]];
				y[j++]=m[s[i++]];
			}
			while(i<s.length);
		}

		@Override
		boolean simd()
		{
			return true;
		}

		private Queue <int[]> transpose(final List <int[]> l, final int vectors, final int lanes)
		{
			final Queue <int[]> q=new ArrayDeque <>(vectors);
			{
				int i=0, j=i;
				final int[] a=new int[lanes];
				do
				{
					a[i]=l.get(i++)[j];
					if(i==lanes)
					{
						if(++j==vectors)
						{
							q.add(a);
							break;
						}
						else
							q.add(a.clone());
						i=0;
					}
				}
				while(true);
			}
			return q;
		}
	}

	private static final class SimdScalar extends Compressor
	{
		private SimdScalar()
		{
			super();
		}

		@Override
		public Compressor clone()
		{
			return this;
		}

		@Override
		Collection <int[]> compress(final List <ChunkState> chunks, final int start, final int end)
		{
			return Blake3.SIMD.compress(chunks,start,end);
		}

		@Override
		int[] function(final int[] state, final int[] words, final int[] chainingValue)
		{
			/*
			 * Uncomment this to compress blocks with scalar compressor and chunks with SIMD. Currently it's always
			 * faster to compress blocks and chunks in sequence with SIMD, so this is disabled. If intrinsics bring
			 * a speed up to scalar compression someday in the future, then this could be useful to enable.
			 * if((state[15]&Blake3.CHUNK_END)!=Blake3.CHUNK_END)
			 * return Blake3.SCALAR.function(state,words,chainingValue);
			 */
			return Blake3.SIMD.function(state,words,chainingValue);
		}

		@Override
		int parallelism()
		{
			/*
			 * It is faster not to use parallel threads at all, but we respect the configuration of the implementation
			 * and use parallel threads if they are enabled.
			 */
			return Blake3.THREADS_ENABLED?Blake3.SIMD.simd()?Simd.P:Scalar.P:1;
		}

		@Override
		boolean simd()
		{
			return Blake3.SIMD.simd();
		}
	}

	private static final class ThreadPool implements Runnable, Closeable, AutoCloseable
	{
		private static final int THREADS=Math.max(1,Runtime.getRuntime().availableProcessors()/2-1);
		private static final ThreadFactory MANAGER=Thread.ofVirtual().name("Blake3-manager").factory();
		private static final ThreadFactory WORKERS=Thread.ofVirtual().name("Blake3-worker-",1).factory();
		private final Set <Reference <Object>> h;
		private final Object lock;
		private ExecutorService m;
		private ExecutorService e;
		private boolean refresh;

		private ThreadPool()
		{
			super();
			h=new HashSet <>();
			lock=new Object();
			m=e=null;
			refresh=false;
		}

		@Override
		public void close()
		{
			synchronized(lock)
			{
				if(h.isEmpty())
				{
					if(refresh)
						return;
					if(e!=null)
					{
						e.close();
						e=null;
					}
					if(m!=null)
					{
						m.close();
						m=null;
					}
				}
			}
		}

		private ExecutorService get(final Object ref, final int threads)
		{
			synchronized(lock)
			{
				refresh=h.add(new WeakReference <>(ref));
				if(e!=null)
					return e;
				if(m==null)
				{
					final long delay=10;
					final ScheduledThreadPoolExecutor init=new ScheduledThreadPoolExecutor(1,ThreadPool.MANAGER);
					init.scheduleAtFixedRate(() ->
					{
						maintain();
					},delay,delay,TimeUnit.SECONDS);
					m=init;
				}
				return e=newPool(threads);
			}
		}

		private void maintain()
		{
			synchronized(lock)
			{
				if(!h.isEmpty())
				{
					for(final Iterator <Reference <Object>> i=h.iterator(); i.hasNext();)
					{
						if(i.next().get()==null)
							i.remove();
					}
				}
				if(refresh)
					refresh=false;
				else
					if(h.isEmpty())
						ThreadPool.MANAGER.newThread(this).start();
			}
		}

		private ExecutorService newPool(final int threads)
		{
			final ThreadPoolExecutor t;
			{
				final long timeout=10;
				final int n=Math.min(threads,ThreadPool.THREADS);
				if(n<1)
					t=new ThreadPoolExecutor(0,Integer.MAX_VALUE,timeout,TimeUnit.SECONDS,new SynchronousQueue <Runnable>(true),ThreadPool.WORKERS);
				else
				{
					t=new ThreadPoolExecutor(n,n,timeout,TimeUnit.SECONDS,new LinkedBlockingQueue <Runnable>(),ThreadPool.WORKERS);
					t.allowCoreThreadTimeOut(true);
					t.prestartAllCoreThreads();
				}
			}
			return t;
		}

		private boolean remove(final Object ref)
		{
			synchronized(lock)
			{
				if(!h.isEmpty())
				{
					for(final Iterator <Reference <Object>> i=h.iterator(); i.hasNext();)
					{
						final Object o=i.next().get();
						if(o==ref)
						{
							i.remove();
							return refresh=true;
						}
						if(o==null)
							i.remove();
					}
				}
			}
			return false;
		}

		@Override
		public void run()
		{
			close();
		}
	}

	/**
	 * A simple timer that can be used in fast and dirty profiling. This is not thread-safe and is only usable during a
	 * single thread execution.
	 * <p>
	 * Example:
	 * <p>
	 * <code>
	 * &nbsp;&nbsp;&nbsp;&nbsp;// first make a static timer:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;private static final Timer TIMER=new Timer();<br><br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;// then in ChunkState::update method:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;{<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Blake3.TIMER.start();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c.compress(...);<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Blake3.TIMER.add();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;}<br><br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;// ...later in Blake3::digest method:<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;System.out.println(getAlgorithm()+": "+Blake3.TIMER.toString());
	 * </code>
	 * <p>
	 * This example would print a string with the total execution time of the compressor method in millisecond
	 * precision.
	 */
	static final class Timer
	{
		private static final String F="%dms";
		private long t, t0;

		Timer()
		{
			super();
			t=t0=0;
		}

		void add()
		{
			t+=System.nanoTime()-t0;
		}

		void start()
		{
			t0=System.nanoTime();
		}

		long stop()
		{
			final long l=t/1000000;
			t=t0=0;
			return l;
		}

		@Override
		public String toString()
		{
			return String.format(Timer.F,stop());
		}
	}

	private static final int CHAIN_LEN=8;
	private static final int WORDS_LEN=16;
	private static final int KEY_LEN=32;
	private static final int STACK_LEN=54;
	private static final int BLOCK_LEN=64;
	private static final int CHUNK_LEN=1024;
	private static final int CHUNK_START=1;
	private static final int CHUNK_END=2;
	private static final int PARENT=4;
	private static final int ROOT=8;
	private static final int KEYED_HASH=16;
	private static final int DERIVE_KEY_CONTEXT=32;
	private static final int DERIVE_KEY_MATERIAL=64;
	private static final int[] IV={0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19};
	private static final byte[] EMPTY=new byte[0];
	private static final boolean THREADS_ENABLED=Blake3.checkParallelism();
	private static final boolean SIMD_ENABLED=Blake3.checkSIMD();
	private static final Compressor SCALAR=Blake3.compressor(false);
	private static final Compressor SIMD=Blake3.compressor(true);
	private static final Compressor OPTIMAL=new SimdScalar();
	private static final ThreadPool POOL=new ThreadPool();
	/**
	 * This constant sets the threshold when parallel compression is used. This value times chunk size is the size in
	 * bytes of data that will be compressed by parallel threads. Currently it is 256*1024 = 262144 bytes. This value
	 * <i>must</i> be a power of two in the range [2,2<sup>n</sup>] where <code>n</code> is an integer > 1.
	 */
	private static final int TASKS=256;
	private static final int TASKS_1=Blake3.TASKS-1;
	/**
	 * This algorithm is allowed to choose SIMD or scalar compressor when it is beneficial. It could compress blocks
	 * with a scalar compressor and chunks with SIMD etc.
	 */
	private static final String ALGORITHM="BLAKE3";
	/**
	 * This algorithm must always use a scalar compressor.
	 */
	private static final String ALGORITHM_SCALAR="BLAKE3-SCALAR";
	/**
	 * This algorithm must always use SIMD if available and a scalar compressor if not.
	 */
	private static final String ALGORITHM_SIMD="BLAKE3-SIMD";

	private static boolean checkParallelism()
	{
		return !System.getProperty(Blake3.class.getCanonicalName()+".threads","true").equalsIgnoreCase("false");
	}

	private static boolean checkSIMD()
	{
		try
		{
			IntVector.SPECIES_PREFERRED.length();
		}
		catch(final Throwable t)
		{
			return false;
		}
		return !System.getProperty(Blake3.class.getCanonicalName()+".simd","true").equalsIgnoreCase("false");
	}

	private static Compressor compressor(final boolean simd)
	{
		if(simd)
		{
			if(Blake3.SIMD_ENABLED)
			{
				try
				{
					return new Simd();
				}
				catch(final Throwable t)
				{}
			}
			if(Blake3.SCALAR!=null)
				return Blake3.SCALAR;
		}
		return new Scalar();
	}

	/**
	 * Returns a new provider for Blake3 message digest instances. Providers for MAC or KDF types are not supported,
	 * since they have to be trusted by JCE to work.
	 * <p>
	 * Supported algorithm names:<br>
	 * <ul>
	 * <li><code>BLAKE3</code> chooses between SIMD and scalar compression if needed
	 * <li><code>BLAKE3-SCALAR</code> uses scalar compression
	 * <li><code>BLAKE3-SIMD</code> uses SIMD if possible and falls back to scalar compression if not
	 * </ul>
	 * <p>
	 * Example:
	 * <p>
	 * <code>
	 * &nbsp;&nbsp;&nbsp;&nbsp;Provider p=Blake3.provider();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;MessageDigest d1=MessageDigest.getInstance("BLAKE3",p);<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;d1.update("Hello there".getBytes(StandardCharsets.UTF_8));<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;MessageDigest d2=(MessageDigest)d1.clone();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;d2.update(" world!".getBytes(StandardCharsets.UTF_8));<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;byte[] hash1=d1.digest();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;byte[] hash2=d2.digest();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;d1.update("Hello again!".getBytes(StandardCharsets.UTF_8));<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;byte[] hash3=d1.digest();
	 * </code>
	 * <p>
	 * <b>Note:</b> {@link Blake3#digest(int) digest} method should <i>always</i> be called, preferably in a finally
	 * block, when Blake3 instances are no longer needed to make sure internal thread pool is properly closed
	 *
	 * @return a new provider for Blake3 message digest instances
	 */
	public static Provider provider()
	{
		return new Blake3Provider();
	}

	private static boolean simd(final String algorithm) throws NoSuchAlgorithmException
	{
		switch(algorithm)
		{
			case ALGORITHM:
				return Blake3.SIMD_ENABLED;
			case ALGORITHM_SCALAR:
				return false;
			case ALGORITHM_SIMD:
				return true;
			default:
				throw new NoSuchAlgorithmException("unknown algorithm: "+algorithm);
		}
	}

	private static int[] words(final byte[] bytes, final int[] words)
	{
		for(int i=0, j=i; i<words.length; i++)
			words[i]=(bytes[j++]&255)|((bytes[j++]&255)<<8)|((bytes[j++]&255)<<16)|((bytes[j++]&255)<<24);
		return words;
	}

	private static int[] wordsChecked(final byte[] bytes)
	{
		if(bytes.length!=Blake3.KEY_LEN)
			throw new IllegalArgumentException("key must be exactly 32 bytes");
		return Blake3.words(bytes,new int[Blake3.CHAIN_LEN]);
	}

	private final Compressor compr;
	private final ChunkState state;
	private final byte[] one;
	private final int[] key;
	private final int[][] stack;
	private int index;
	private final int flags;
	private final int lanes;
	private final List <ChunkState> q;
	private ExecutorService pool=null;

	/**
	 * Constructs a new hasher for the regular hash function.
	 */
	public Blake3()
	{
		this(Blake3.SIMD_ENABLED);
	}

	/**
	 * Constructs a clone of the hasher.
	 *
	 * @param b
	 *            the hasher to clone
	 */
	private Blake3(final Blake3 b)
	{
		super(b.getAlgorithm());
		compr=b.compr.clone();
		{
			lanes=b.lanes;
			if(lanes>1)
			{
				q=new ArrayList <>(Blake3.TASKS);
				if(!b.q.isEmpty())
					b.q.stream().forEach(e -> q.add(e.clone()));
			}
			else
				q=null;
		}
		state=b.state.clone();
		one=new byte[1];
		key=b.key.clone();
		stack=b.stack.clone();
		index=b.index;
		flags=b.flags;
	}

	/**
	 * Constructs a new hasher for the regular hash function.
	 *
	 * @param simd
	 *            <code>true</code> to use SIMD vectorization
	 */
	public Blake3(final boolean simd)
	{
		this(Blake3.IV,0,simd);
	}

	/**
	 * Constructs a new hasher for the keyed hash function.
	 * <p>
	 * <b>Note:</b> the key must not be modified, because it is used directly by this instance
	 *
	 * @param key
	 *            hash key
	 */
	public Blake3(final byte[] key)
	{
		this(key,Blake3.SIMD_ENABLED);
	}

	/**
	 * Constructs a new hasher for the keyed hash function.
	 * <p>
	 * <b>Note:</b> the key must not be modified, because it is used directly by this instance
	 *
	 * @param key
	 *            hash key
	 * @param simd
	 *            <code>true</code> to use SIMD vectorization
	 */
	public Blake3(final byte[] key, final boolean simd)
	{
		this(Blake3.wordsChecked(key),Blake3.KEYED_HASH,simd);
	}

	/**
	 * Constructs a new hasher.
	 * <p>
	 * <b>Note:</b> the key must not be modified, because it is used directly by this instance
	 *
	 * @param key
	 *            hash key
	 * @param flags
	 *            the flags to use
	 * @param simd
	 *            <code>true</code> to use SIMD vectorization
	 */
	private Blake3(final int[] key, final int flags, final boolean simd)
	{
		this(Blake3.ALGORITHM,key,flags,simd);
	}

	/**
	 * Constructs a new hasher with the specified algorithm name. It is usually <i>not correct</i> to call this case
	 * sensitive constructor directly, but it supports third-party providers that want to use instances of this class in
	 * their services.
	 *
	 * @param algorithm
	 *            the case sensitive standard name of the algorithm
	 * @throws NoSuchAlgorithmException
	 *             if no Blake3 {@link Blake3#provider() Provider} supports a {@link MessageDigestSpi} implementation
	 *             for the specified case sensitive algorithm
	 */
	public Blake3(final String algorithm) throws NoSuchAlgorithmException
	{
		this(algorithm,Blake3.IV,0,Blake3.simd(algorithm));
	}

	/**
	 * Constructs a new hasher for the key derivation function. The context string should be hard coded, globally
	 * unique, and application-specific.
	 *
	 * @param context
	 *            context string
	 * @param charset
	 *            the character set to use for context
	 */
	public Blake3(final String context, final Charset charset)
	{
		this(context,charset,Blake3.SIMD_ENABLED);
	}

	/**
	 * Constructs a new hasher for the key derivation function. The context string should be hard coded, globally
	 * unique, and application-specific.
	 *
	 * @param context
	 *            context string
	 * @param charset
	 *            the character set to use for context
	 * @param simd
	 *            <code>true</code> to use SIMD vectorization
	 */
	public Blake3(final String context, final Charset charset, final boolean simd)
	{
		this(Blake3.words(new Blake3(Blake3.IV,Blake3.DERIVE_KEY_CONTEXT,simd).context(context,charset),new int[Blake3.CHAIN_LEN]),Blake3.DERIVE_KEY_MATERIAL,simd);
	}

	/**
	 * Constructs a new hasher.
	 * <p>
	 * <b>Note:</b> the key must not be modified, because it is used directly by this instance
	 *
	 * @param algorithm
	 *            the algorithm to use
	 * @param key
	 *            hash key
	 * @param flags
	 *            the flags to use
	 * @param simd
	 *            <code>true</code> to use SIMD vectorization
	 */
	private Blake3(final String algorithm, final int[] key, final int flags, final boolean simd)
	{
		super(algorithm);
		compr=Blake3.ALGORITHM.equals(algorithm)?Blake3.OPTIMAL:simd?Blake3.SIMD:Blake3.SCALAR;
		{
			lanes=compr.parallelism();
			q=lanes>1?new ArrayList <>(Blake3.TASKS):null;
		}
		state=new ChunkState(key,flags);
		one=new byte[1];
		this.key=key;
		stack=new int[Blake3.STACK_LEN][];
		index=0;
		this.flags=flags;
	}

	private void addChunk(final ChunkState c)
	{
		if(lanes==1)
		{
			if(c==null)
				return;
			addChunkSerial(c,c.output().chainingValue(compr));
		}
		else
			addChunkParallel(c);
		if(c!=null)
			state.reset(key);
	}

	private void addChunkParallel(final ChunkState c)
	{
		if((c!=null)&&(queue(c)))
			state.counter++;
		else
			if(!q.isEmpty())
			{
				try
				{
					final Queue <int[]> r=compress();
					final int len=q.size();
					int i=0;
					do
					{
						addChunkSerial(q.get(i),r.poll());
					}
					while(++i<len);
				}
				catch(final RuntimeException e)
				{
					throw e;
				}
				catch(final Exception e)
				{
					throw new RuntimeException(e);
				}
				finally
				{
					q.clear();
				}
			}
	}

	private void addChunkSerial(final ChunkState c, int[] chainingValue)
	{
		long counter=++c.counter;
		while((counter&1)==0)
		{
			chainingValue=c.out.parentOutput(c.out.state,stack[--index],chainingValue,key,flags).chainingValue(compr);
			counter>>=1;
		}
		stack[index++]=c==state?chainingValue.clone():chainingValue;
	}

	/**
	 * Clones this instance. The clone inherits the current state of this instance.
	 *
	 * @return a clone of this instance
	 */
	@Override
	public Blake3 clone()
	{
		return new Blake3(this);
	}

	private void close()
	{
		pool=null;
		engineReset();
		Blake3.POOL.remove(this);
	}

	private Queue <int[]> compress() throws Exception
	{
		int i=0;
		final int len=q.size();
		final Queue <int[]> r=new ArrayDeque <>(len);
		if((!Blake3.THREADS_ENABLED)||(len<Blake3.TASKS))
		{
			if((compr.simd())&&(len>=lanes))
			{
				int j;
				do
				{
					j=i+lanes;
					r.addAll(compr.compress(q,i,j));
					i=j;
				}
				while(i+lanes<len);
			}
			while(i<len)
				r.add(q.get(i++).output().chainingValue(compr));
			return r;
		}
		/*
		 * Here we know that there are exactly Blake3.TASKS number of chunks and a total of 262144 bytes to compress in
		 * parallel. The last chunks are compressed by this thread, so that's why we subtract lanes from len to get the
		 * number of chunks we send to thread pool.
		 */
		final int n=len-lanes;
		final Queue <Future <Collection <int[]>>> p=new ArrayDeque <>(n/lanes);
		{
			final ExecutorService e=pool!=null?pool:(pool=Blake3.POOL.get(this,n/lanes));
			do
			{
				final int start=i;
				i+=lanes;
				final int end=i;
				p.add(e.submit(new Callable <Collection <int[]>>()
				{
					@Override
					public Collection <int[]> call()
					{
						return compr.compress(q,start,end);
					}
				}));
			}
			while(i<n);
		}
		final Collection <int[]> tail=compr.compress(q,i,len);
		Future <Collection <int[]>> f;
		while((f=p.poll())!=null)
			r.addAll(f.get());
		r.addAll(tail);
		return r;
	}

	private byte[] context(final String context, final Charset charset)
	{
		final byte[] ctx=context.getBytes(charset);
		engineUpdate(ctx,0,ctx.length);
		return digest(Blake3.KEY_LEN);
	}

	/**
	 * Completes the hash computation and writes any number of output bytes. The digest is reset after this call is
	 * made. Final use of this instance should <i>always</i> be a digest method call to make sure internal thread pool
	 * is properly closed. If digest is no longer needed in such a case then <code>digest(0)</code> is fast.
	 *
	 * @param len
	 *            hash length in bytes
	 * @return an array of bytes for the resulting hash value
	 */
	public byte[] digest(final int len)
	{
		try
		{
			return len==0?Blake3.EMPTY:finalOutput().rootBytes(new byte[len],0,len,compr);
		}
		finally
		{
			close();
		}
	}

	@Override
	protected byte[] engineDigest()
	{
		return digest(Blake3.KEY_LEN);
	}

	@Override
	protected int engineDigest(final byte[] buf, final int offset, final int len) throws DigestException
	{
		final byte[] b=digest(len);
		final int limit=Math.addExact(offset,len);
		for(int i=offset, i2=0; i<limit; i++, i2++)
			buf[i]=b[i2];
		return b.length;
	}

	@Override
	protected void engineReset()
	{
		index=0;
		Arrays.fill(stack,null);
		state.clear(key);
		if(q!=null)
			q.clear();
	}

	@Override
	protected void engineUpdate(final byte input)
	{
		one[0]=input;
		engineUpdate(one,0,1);
	}

	@Override
	protected void engineUpdate(final byte[] input, int offset, final int len)
	{
		try
		{
			final int limit=Math.addExact(offset,len);
			while(offset<limit)
			{
				if(state.length()==Blake3.CHUNK_LEN)
					addChunk(state);
				state.update(input,offset,offset+=Math.min(Blake3.CHUNK_LEN-state.length(),limit-offset),compr);
			}
		}
		catch(final Throwable t)
		{
			close();
			throw t;
		}
	}

	@Override
	protected void engineUpdate(final ByteBuffer input)
	{
		final int len=input.remaining();
		if(len>0)
		{
			if(input.hasArray())
			{
				final int p=input.position();
				update(input.array(),p,len);
				input.position(p+len);
				return;
			}
			final byte[] b=new byte[len];
			input.get(b,0,len);
			update(b,0,len);
		}
	}

	private Output finalOutput()
	{
		// this makes sure there are no chunks in queue
		addChunk(null);
		final Output o=state.output();
		while(index>0)
			o.parentOutput(o.state,stack[--index],o.chainingValue(compr),key,flags);
		return o;
	}

	byte[] hash(final String filename, final int len) throws IOException
	{
		/*
		 * The purpose of this method was to test is reading optimal chunks of data beneficial for parallel computation.
		 * This should not be a public method, because it can cause buffer overflows and invalid hash values if for some
		 * reason it does not read exactly 8192 bytes of large files per loop.
		 */
		try(FileChannel c=FileChannel.open(Paths.get(filename),StandardOpenOption.READ))
		{
			/*
			 * chunk*n must be equal to Blake3.TASKS*Blake3.CHUNK_LEN
			 */
			int i, j=0;
			final int n=32;
			final int chunk=8192;
			final ByteBuffer b=ByteBuffer.allocateDirect(chunk);
			final ByteBuffer l=ByteBuffer.allocateDirect(chunk*n);
			while((i=c.read(b))>0)
			{
				l.put(b.flip());
				if((i==chunk)&&(++j==n))
				{
					engineUpdate(l.flip());
					l.clear();
					j=0;
				}
				b.clear();
			}
			if(l.flip().hasRemaining())
				engineUpdate(l);
		}
		return digest(len);
	}

	private boolean queue(final ChunkState c)
	{
		return q.size()<Blake3.TASKS_1?q.add(c.clone()):!q.add(c);
	}

	/**
	 * Returns <code>true</code> if this instance uses SIMD vectorization.
	 *
	 * @return <code>true</code> if this instance uses SIMD vectorization
	 */
	public boolean simd()
	{
		return compr.simd();
	}
}
