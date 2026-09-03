/*
 * @formatter:off
 *
 * Copyright 2026 Archon Research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @formatter:on
 */
package com.eternitymud.core.util;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.security.DigestException;
import java.security.MessageDigest;
import java.security.MessageDigestSpi;
import java.security.NoSuchAlgorithmException;
import java.security.Provider;
import java.security.Provider.Service;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Queue;
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
 * This class implements two levels of SIMD (Single Instruction Multiple Data) parallelism that are discussed in section
 * 5.3 of BLAKE3 <a href="https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf">whitepaper</a>: full-width
 * lane-parallel compression of chunk batches, and fixed-width 128-bit vectors for individual block compression outside
 * those batches (remainders, parent nodes, root output). SIMD performance depends on the AVX/AVX-512/NEON etc.
 * capabilities of the hardware. Scalar compressor is used {@link Blake3#simd() automatically} if the vectorization API
 * is unavailable or SIMD is disabled.
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
 * &nbsp;&nbsp;&nbsp;&nbsp;b1.update("Hello there".getBytes());<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;Blake3 b2=b1.clone();<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;b2.update(" world!".getBytes());<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;System.out.println(Base64.getEncoder().encodeToString(b1.digest()));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;-&gt; 9FQM6IvkXlqP1tGY/5Dax+s03Yx51t+4dnoXALEIrm8=<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;System.out.println(Base64.getEncoder().encodeToString(b2.digest()));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;-&gt; 6XeH1EeHk0/YYn+x73TYgOHk9Y+ibl8Z0d56OOXH4mk=<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;b1.update("Hello again!".getBytes());<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;System.out.println(Base64.getEncoder().encodeToString(b1.digest()));<br>
 * &nbsp;&nbsp;&nbsp;&nbsp;-&gt; PyX4XKZIhebsMQd88DHHBUrMnl99D7HzdblCEFtcxAs=
 * </code>
 * <p>
 * There is a system property that can be used to configure Blake3 instances:
 * <p>
 * <code>
 * &nbsp;&nbsp;&nbsp;&nbsp;com.eternitymud.core.util.Blake3.simd=true<br>
 * </code>
 * <p>
 * If that property is set to <code>false</code> then SIMD vectorization is disabled. The property must be set before
 * this class is loaded, usually when Java virtual machine is started.
 * <p>
 * This implementation has been tested with the
 * <a href="https://github.com/BLAKE3-team/BLAKE3/blob/master/test_vectors/test_vectors.json">vectors</a> of the
 * official BLAKE3 repository for all types (regular hash, keyed hash, key derivation) with and without SIMD
 * vectorization. Numerous files of different sizes have also been hashed and the output matched to
 * <a href="https://crates.io/crates/b3sum"><code>b3sum</code></a> utility.
 *
 * @version 2.0
 * @author Zan the Archon &lt;zan@fantasymail.de&gt;
 */
public final class Blake3 extends MessageDigest implements Cloneable
{
	private static final class Blake3Provider extends Provider
	{
		private static final long serialVersionUID=1L;

		private Blake3Provider()
		{
			super("Blake3Provider","2.0","Blake3Provider - Blake3 message digest provider");
			final String type="MessageDigest";
			final String name=Blake3.class.getCanonicalName();
			putService(new Blake3Service(this,type,Blake3.ALGORITHM,name,null,null));
			putService(new Blake3Service(this,type,Blake3.ALGORITHM_SCALAR,name,null,null));
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

		abstract int[] function(int[] state, int[] words, int[] chainingValue);

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
		boolean simd()
		{
			return false;
		}
	}

	private static final class Simd extends Compressor
	{
		private static final VectorSpecies <Integer> SPEC=IntVector.SPECIES_128;
		private static final VectorShuffle <Integer> S1=VectorShuffle.fromArray(Simd.SPEC,new int[]{1,2,3,0},0);
		private static final VectorShuffle <Integer> S2=VectorShuffle.fromArray(Simd.SPEC,new int[]{2,3,0,1},0);
		private static final VectorShuffle <Integer> S3=VectorShuffle.fromArray(Simd.SPEC,new int[]{3,0,1,2},0);

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
	}

	private static final class Wide
	{
		private static final VectorSpecies <Integer> PREF=IntVector.SPECIES_PREFERRED;
		private static final int LANES=Wide.PREF.length();

		private static Queue <int[]> hashBatch(final byte[] in, final int off, final long baseChunk, final int[] key, final int flags)
		{
			/*
			 * Java vectors are value-based classes, and cannot be used as array elements or method parameters without
			 * extreme performance penalties. All compressor rounds and the mixer function G are done directly in this
			 * method because of that property.
			 */
			int i, j, k;
			final int[] lo=new int[Wide.LANES];
			final int[] hi=new int[Wide.LANES];
			{
				long l=baseChunk;
				for(i=0; i<Wide.LANES; i++, l++)
				{
					lo[i]=(int)l;
					hi[i]=(int)(l>>>32);
				}
			}
			final int[] ints=new int[Wide.LANES*(Blake3.CHUNK_LEN/Integer.BYTES)];
			{
				ByteBuffer.wrap(in,off,Wide.LANES*Blake3.CHUNK_LEN).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(ints);
			}
			final int[][] m=new int[Blake3.WORDS_LEN][Wide.LANES];
			IntVector v0=IntVector.broadcast(Wide.PREF,key[0]);
			IntVector v1=IntVector.broadcast(Wide.PREF,key[1]);
			IntVector v2=IntVector.broadcast(Wide.PREF,key[2]);
			IntVector v3=IntVector.broadcast(Wide.PREF,key[3]);
			IntVector v4=IntVector.broadcast(Wide.PREF,key[4]);
			IntVector v5=IntVector.broadcast(Wide.PREF,key[5]);
			IntVector v6=IntVector.broadcast(Wide.PREF,key[6]);
			IntVector v7=IntVector.broadcast(Wide.PREF,key[7]);
			IntVector a, b, c, d;
			i=0;
			do
			{
				IntVector v8=IntVector.broadcast(Wide.PREF,Blake3.IV[0]);
				IntVector v9=IntVector.broadcast(Wide.PREF,Blake3.IV[1]);
				IntVector v10=IntVector.broadcast(Wide.PREF,Blake3.IV[2]);
				IntVector v11=IntVector.broadcast(Wide.PREF,Blake3.IV[3]);
				IntVector v12=IntVector.fromArray(Wide.PREF,lo,0);
				IntVector v13=IntVector.fromArray(Wide.PREF,hi,0);
				IntVector v14=IntVector.broadcast(Wide.PREF,Blake3.BLOCK_LEN);
				IntVector v15=IntVector.broadcast(Wide.PREF,flags|(i==0?Blake3.CHUNK_START:0)|(i==15?Blake3.CHUNK_END:0));
				{
					final int z=i*Blake3.WORDS_LEN;
					for(j=0; j<Blake3.WORDS_LEN; j++)
					{
						for(k=0; k<Wide.LANES; k++)
							m[j][k]=ints[z+k*(Blake3.CHUNK_LEN/Integer.BYTES)+j];
					}
				}
				j=0;
				do
				{
					final int[] s=Compressor.SCHEDULE[j];
					final IntVector m0=IntVector.fromArray(Wide.PREF,m[s[0]],0);
					final IntVector m1=IntVector.fromArray(Wide.PREF,m[s[1]],0);
					final IntVector m2=IntVector.fromArray(Wide.PREF,m[s[2]],0);
					final IntVector m3=IntVector.fromArray(Wide.PREF,m[s[3]],0);
					final IntVector m4=IntVector.fromArray(Wide.PREF,m[s[4]],0);
					final IntVector m5=IntVector.fromArray(Wide.PREF,m[s[5]],0);
					final IntVector m6=IntVector.fromArray(Wide.PREF,m[s[6]],0);
					final IntVector m7=IntVector.fromArray(Wide.PREF,m[s[7]],0);
					final IntVector m8=IntVector.fromArray(Wide.PREF,m[s[8]],0);
					final IntVector m9=IntVector.fromArray(Wide.PREF,m[s[9]],0);
					final IntVector m10=IntVector.fromArray(Wide.PREF,m[s[10]],0);
					final IntVector m11=IntVector.fromArray(Wide.PREF,m[s[11]],0);
					final IntVector m12=IntVector.fromArray(Wide.PREF,m[s[12]],0);
					final IntVector m13=IntVector.fromArray(Wide.PREF,m[s[13]],0);
					final IntVector m14=IntVector.fromArray(Wide.PREF,m[s[14]],0);
					final IntVector m15=IntVector.fromArray(Wide.PREF,m[s[15]],0);
					// Column G1: G(v0,v4,v8,v12,m0,m1)
					a=v0.add(v4).add(m0);
					d=v12.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,16).or(d.lanewise(VectorOperators.LSHL,16));
					c=v8.add(d);
					b=v4.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,12).or(b.lanewise(VectorOperators.LSHL,20));
					a=a.add(b).add(m1);
					d=d.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,8).or(d.lanewise(VectorOperators.LSHL,24));
					c=c.add(d);
					b=b.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,7).or(b.lanewise(VectorOperators.LSHL,25));
					v0=a;
					v4=b;
					v8=c;
					v12=d;
					// Column G2: G(v1,v5,v9,v13,m2,m3)
					a=v1.add(v5).add(m2);
					d=v13.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,16).or(d.lanewise(VectorOperators.LSHL,16));
					c=v9.add(d);
					b=v5.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,12).or(b.lanewise(VectorOperators.LSHL,20));
					a=a.add(b).add(m3);
					d=d.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,8).or(d.lanewise(VectorOperators.LSHL,24));
					c=c.add(d);
					b=b.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,7).or(b.lanewise(VectorOperators.LSHL,25));
					v1=a;
					v5=b;
					v9=c;
					v13=d;
					// Column G3: G(v2,v6,v10,v14,m4,m5)
					a=v2.add(v6).add(m4);
					d=v14.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,16).or(d.lanewise(VectorOperators.LSHL,16));
					c=v10.add(d);
					b=v6.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,12).or(b.lanewise(VectorOperators.LSHL,20));
					a=a.add(b).add(m5);
					d=d.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,8).or(d.lanewise(VectorOperators.LSHL,24));
					c=c.add(d);
					b=b.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,7).or(b.lanewise(VectorOperators.LSHL,25));
					v2=a;
					v6=b;
					v10=c;
					v14=d;
					// Column G4: G(v3,v7,v11,v15,m6,m7)
					a=v3.add(v7).add(m6);
					d=v15.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,16).or(d.lanewise(VectorOperators.LSHL,16));
					c=v11.add(d);
					b=v7.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,12).or(b.lanewise(VectorOperators.LSHL,20));
					a=a.add(b).add(m7);
					d=d.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,8).or(d.lanewise(VectorOperators.LSHL,24));
					c=c.add(d);
					b=b.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,7).or(b.lanewise(VectorOperators.LSHL,25));
					v3=a;
					v7=b;
					v11=c;
					v15=d;
					// Diagonal G5: G(v0,v5,v10,v15,m8,m9)
					a=v0.add(v5).add(m8);
					d=v15.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,16).or(d.lanewise(VectorOperators.LSHL,16));
					c=v10.add(d);
					b=v5.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,12).or(b.lanewise(VectorOperators.LSHL,20));
					a=a.add(b).add(m9);
					d=d.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,8).or(d.lanewise(VectorOperators.LSHL,24));
					c=c.add(d);
					b=b.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,7).or(b.lanewise(VectorOperators.LSHL,25));
					v0=a;
					v5=b;
					v10=c;
					v15=d;
					// Diagonal G6: G(v1,v6,v11,v12,m10,m11)
					a=v1.add(v6).add(m10);
					d=v12.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,16).or(d.lanewise(VectorOperators.LSHL,16));
					c=v11.add(d);
					b=v6.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,12).or(b.lanewise(VectorOperators.LSHL,20));
					a=a.add(b).add(m11);
					d=d.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,8).or(d.lanewise(VectorOperators.LSHL,24));
					c=c.add(d);
					b=b.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,7).or(b.lanewise(VectorOperators.LSHL,25));
					v1=a;
					v6=b;
					v11=c;
					v12=d;
					// Diagonal G7: G(v2,v7,v8,v13,m12,m13)
					a=v2.add(v7).add(m12);
					d=v13.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,16).or(d.lanewise(VectorOperators.LSHL,16));
					c=v8.add(d);
					b=v7.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,12).or(b.lanewise(VectorOperators.LSHL,20));
					a=a.add(b).add(m13);
					d=d.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,8).or(d.lanewise(VectorOperators.LSHL,24));
					c=c.add(d);
					b=b.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,7).or(b.lanewise(VectorOperators.LSHL,25));
					v2=a;
					v7=b;
					v8=c;
					v13=d;
					// Diagonal G8: G(v3,v4,v9,v14,m14,m15)
					a=v3.add(v4).add(m14);
					d=v14.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,16).or(d.lanewise(VectorOperators.LSHL,16));
					c=v9.add(d);
					b=v4.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,12).or(b.lanewise(VectorOperators.LSHL,20));
					a=a.add(b).add(m15);
					d=d.lanewise(VectorOperators.XOR,a);
					d=d.lanewise(VectorOperators.LSHR,8).or(d.lanewise(VectorOperators.LSHL,24));
					c=c.add(d);
					b=b.lanewise(VectorOperators.XOR,c);
					b=b.lanewise(VectorOperators.LSHR,7).or(b.lanewise(VectorOperators.LSHL,25));
					v3=a;
					v4=b;
					v9=c;
					v14=d;
				}
				while(++j<Compressor.ROUNDS);
				v0=v0.lanewise(VectorOperators.XOR,v8);
				v1=v1.lanewise(VectorOperators.XOR,v9);
				v2=v2.lanewise(VectorOperators.XOR,v10);
				v3=v3.lanewise(VectorOperators.XOR,v11);
				v4=v4.lanewise(VectorOperators.XOR,v12);
				v5=v5.lanewise(VectorOperators.XOR,v13);
				v6=v6.lanewise(VectorOperators.XOR,v14);
				v7=v7.lanewise(VectorOperators.XOR,v15);
			}
			while(++i<Blake3.CHUNK_LEN/Blake3.BLOCK_LEN);
			final int[][] rows=new int[Blake3.CHAIN_LEN][];
			rows[0]=v0.toArray();
			rows[1]=v1.toArray();
			rows[2]=v2.toArray();
			rows[3]=v3.toArray();
			rows[4]=v4.toArray();
			rows[5]=v5.toArray();
			rows[6]=v6.toArray();
			rows[7]=v7.toArray();
			return Wide.values(rows);
		}

		private static Queue <int[]> values(final int[][] rows)
		{
			final Queue <int[]> q=new ArrayDeque <>(Wide.LANES);
			for(int i=0; i<Wide.LANES; i++)
			{
				final int[] a=new int[Blake3.WORDS_LEN];
				for(int j=0; j<Blake3.CHAIN_LEN; j++)
					a[j]=rows[j][i];
				q.add(a);
			}
			return q;
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
	private static final boolean SIMD_ENABLED=Blake3.checkSIMD();
	private static final Compressor SCALAR=Blake3.compressor(false);
	private static final Compressor SIMD=Blake3.compressor(true);
	/**
	 * This algorithm uses the wide chunk pipeline with SIMD block compression if available and a scalar compressor if
	 * not.
	 */
	private static final String ALGORITHM="BLAKE3";
	/**
	 * This algorithm must always use a scalar compressor.
	 */
	private static final String ALGORITHM_SCALAR="BLAKE3-SCALAR";

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
	 * <li><code>BLAKE3</code> uses SIMD if available and scalar compression if not
	 * <li><code>BLAKE3-SCALAR</code> uses scalar compression
	 * </ul>
	 * <p>
	 * Example:
	 * <p>
	 * <code>
	 * &nbsp;&nbsp;&nbsp;&nbsp;Provider p=Blake3.provider();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;MessageDigest d1=MessageDigest.getInstance("BLAKE3",p);<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;d1.update("Hello there".getBytes());<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;MessageDigest d2=(MessageDigest)d1.clone();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;d2.update(" world!".getBytes());<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;byte[] hash1=d1.digest();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;byte[] hash2=d2.digest();<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;d1.update("Hello again!".getBytes());<br>
	 * &nbsp;&nbsp;&nbsp;&nbsp;byte[] hash3=d1.digest();
	 * </code>
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
	private final byte[] pending;
	private int pendingLen;
	private final boolean wide;

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
		state=b.state.clone();
		one=new byte[1];
		key=b.key.clone();
		stack=b.stack.clone();
		index=b.index;
		flags=b.flags;
		pending=b.pending!=null?b.pending.clone():null;
		pendingLen=b.pendingLen;
		wide=b.wide;
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
		compr=simd?Blake3.SIMD:Blake3.SCALAR;
		state=new ChunkState(key,flags);
		one=new byte[1];
		this.key=key;
		stack=new int[Blake3.STACK_LEN][];
		index=0;
		this.flags=flags;
		wide=simd&&Blake3.SIMD_ENABLED&&Wide.LANES>1;
		pending=wide?new byte[Wide.LANES*Blake3.CHUNK_LEN<<1]:null;
		pendingLen=0;
	}

	private void addChunk()
	{
		int[] chainingValue=state.output().chainingValue(compr);
		long counter=++state.counter;
		while((counter&1)==0)
		{
			chainingValue=state.out.parentOutput(state.out.state,stack[--index],chainingValue,key,flags).chainingValue(compr);
			counter>>=1;
		}
		stack[index++]=chainingValue.clone();
		state.reset(key);
	}

	private void addChunkWide(int[] chainingValue, long counter)
	{
		counter++;
		while((counter&1)==0)
		{
			chainingValue=state.out.parentOutput(state.state,stack[--index],chainingValue,key,flags).chainingValue(compr).clone();
			counter>>=1;
		}
		stack[index++]=chainingValue;
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

	private byte[] context(final String context, final Charset charset)
	{
		final byte[] ctx=context.getBytes(charset);
		engineUpdate(ctx,0,ctx.length);
		return digest(Blake3.KEY_LEN);
	}

	/**
	 * Completes the hash computation and writes any number of output bytes. The digest is reset after this call is
	 * made.
	 *
	 * @param len
	 *            hash length in bytes
	 * @return an array of bytes for the resulting hash value
	 */
	public byte[] digest(final int len)
	{
		try
		{
			if((wide)&&(pendingLen>0))
			{
				updateDirect(pending,0,pendingLen);
				pendingLen=0;
			}
			return len==0?Blake3.EMPTY:finalOutput().rootBytes(new byte[len],0,len,compr);
		}
		finally
		{
			engineReset();
		}
	}

	private void drain()
	{
		final int batch=Wide.LANES*Blake3.CHUNK_LEN;
		if(pendingLen<batch<<1)
			return;
		int i;
		do
		{
			final Queue <int[]> r=Wide.hashBatch(pending,0,state.counter,key,flags);
			for(i=0; i<Wide.LANES; i++)
				addChunkWide(r.poll(),state.counter+i);
			state.counter+=Wide.LANES;
			pendingLen-=batch;
			System.arraycopy(pending,batch,pending,0,pendingLen);
		}
		while(pendingLen>=batch<<1);
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
		index=pendingLen=0;
		Arrays.fill(stack,null);
		state.clear(key);
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
			if(!wide)
			{
				updateDirect(input,offset,limit);
				return;
			}
			int n;
			while(offset<limit)
			{
				n=Math.min(pending.length-pendingLen,limit-offset);
				System.arraycopy(input,offset,pending,pendingLen,n);
				pendingLen+=n;
				offset+=n;
				drain();
			}
		}
		catch(final Throwable t)
		{
			engineReset();
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
				engineUpdate(input.array(),p,len);
				input.position(p+len);
				return;
			}
			final byte[] b=new byte[len];
			input.get(b,0,len);
			engineUpdate(b,0,len);
		}
	}

	private Output finalOutput()
	{
		final Output o=state.output();
		while(index>0)
			o.parentOutput(o.state,stack[--index],o.chainingValue(compr),key,flags);
		return o;
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

	private void updateDirect(final byte[] input, int offset, final int limit)
	{
		while(offset<limit)
		{
			if(state.length()==Blake3.CHUNK_LEN)
				addChunk();
			state.update(input,offset,offset+=Math.min(Blake3.CHUNK_LEN-state.length(),limit-offset),compr);
		}
	}
}
