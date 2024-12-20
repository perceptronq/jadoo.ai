"use client";

import Link from 'next/link';
import Image from 'next/image';
import Logo from '@/assets/logo.png';
import Gif from '@/assets/dhoop-jadu.gif';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';

const SignupPage = () => {
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [showConfirmation, setShowConfirmation] = useState(false);

    const handleSignup = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const { error } = await supabase.auth.signUp({
                email,
                password,
                options: {
                    emailRedirectTo: `${window.location.origin}/auth/callback`,
                }
            });

            if (error) throw error;

            setShowConfirmation(true);
        } catch (error: any) {
            setError(error.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen">
            <div className="flex items-center justify-center">
                <Image
                    src={Logo}
                    alt="Logo"
                    width={50}
                    height={50}
                    className="mr-2 py-4"
                    priority
                />
                <h1 className="text-2xl font-bold mb-0">
                    <Link href="/">
                        Jadoo.ai
                    </Link>
                </h1>
            </div>
            {showConfirmation ? (
                <div className="text-center mt-4 p-4 border border-white/15 rounded-md">
                    <p className="text-white mb-4">
                        Please check your email to confirm your registration.
                    </p>
                    <Link href="/login" className="hover:underline">
                        Back to Login
                    </Link>
                </div>
            ) : (
                <form className="border border-white/15 rounded-md px-8 pt-6 pb-8 mb-4 w-96" onSubmit={handleSignup}>
                    <div className="mb-4">
                        <label className="block text-white text-sm font-bold mb-2" htmlFor="email">
                            Email
                        </label>
                        <input
                            className="shadow appearance-none border border-white/15 rounded-md w-full py-2 px-3 bg-transparent text-white leading-tight focus:outline-none focus:shadow-outline"
                            id="email"
                            type="email"
                            placeholder="enter your email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>
                    <div className="mb-4">
                        <label className="block text-white text-sm font-bold mb-2" htmlFor="password">
                            Password
                        </label>
                        <input
                            className="shadow appearance-none border border-white/15 rounded-md w-full py-2 px-3 bg-transparent text-white leading-tight focus:outline-none focus:shadow-outline"
                            id="password"
                            type="password"
                            placeholder="enter your password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    <div className="flex items-center justify-center py-2">
                        <button
                            className="bg-transparent text-white hover:bg-white/10 border border-white/15 rounded-md font-bold py-2 px-4 focus:outline-none focus:shadow-outline disabled:opacity-50"
                            type="submit"
                            disabled={loading}
                        >
                            {loading ? 'Loading...' : 'Signup'}
                        </button>
                    </div>
                    <p className="text-gray-400 text-sm pt-2 text-center">
                        Already have an account?{' '}
                        <Link href="/login" className="hover:underline">
                            Login
                        </Link>
                    </p>
                </form>
            )}
            <div className="mx-auto py-6">
                <Image
                    src={Gif}
                    alt="Jadoo GIF"
                    width={200}
                    height={200}
                    priority
                    className="mx-auto rounded-md"
                />
            </div>
        </div>
    );
};

export default SignupPage;